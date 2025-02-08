import pinocchio as pin
import numpy as np
from pinocchio import casadi as cpin
import casadi

class RobotKinematics:
    def __init__(self, model_path, end_frame_name, predict_horizon=100):
        """
        Initialize the robot kinematics class.
        
        Args:
            model_path (str): Path to the URDF model file
            end_frame_name (str): Name of the end effector frame
        """
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        self.end_frame_name = end_frame_name
        self.endEffector_ID = self.model.getFrameId(end_frame_name)
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        nq = self.model.nq
        nv = self.model.nv
        nx = nq + nv
        self.cx = casadi.SX.sym("x", nx, 1)
        self.cq = self.cx[:nq]
        self.cv = self.cx[nq:]
        self.caq = casadi.SX.sym("a", nv, 1)

        self.ceef_pos = casadi.SX.sym("eef_pos", 3, 1)
        self.ceef_rot = casadi.SX.sym("eef_rot", 3, 3)

        self.joint_limits = {
            'lower': np.array([-1.2, -0.15, 0, 0]),
            'upper': np.array([2.3, 1, 1, 2]),
        }

        self.velocity_limits = np.array([1.0, 1.0, 1.0, 1.0])  # Example velocity limits
        self._create_error_functions()

    def _create_error_functions(self):
        """Create the 3D and 6D error functions using Casadi"""
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        self.error3_tool = casadi.Function(
            "etool3",
            [self.cq, self.ceef_pos],
            [self.cdata.oMf[self.endEffector_ID].translation - self.ceef_pos]
        )
        self.error6_tool = casadi.Function(
            "etool6",
            [self.cq, self.ceef_pos, self.ceef_rot],
            [cpin.log6(self.cdata.oMf[self.endEffector_ID].inverse() *
             cpin.SE3(self.ceef_rot, self.ceef_pos)).vector]
        )

    def compute_position_error(self, q, target_pos):
        return self.error3_tool(q, target_pos)

    def compute_pose_error(self, q, target_pos, target_rot):
        return self.error6_tool(q, target_pos, target_rot)

    def define_mpc(self, T=100, w_run=0.1, w_term=10.0, DT=0.002, use_pose_error=True):
        opti = casadi.Opti()

        param_x0 = opti.parameter(self.model.nq + self.model.nv)
        param_target_pos = opti.parameter(3)
        param_target_rot = opti.parameter(3, 3)

        var_xs = [opti.variable(self.model.nq + self.model.nv) for t in range(T + 1)]
        var_as = [opti.variable(self.model.nv) for t in range(T)]

        totalcost = 0
        for t in range(T):
            totalcost += w_run * DT * casadi.sumsqr(var_xs[t][self.model.nq:])  # Running cost
            totalcost += 0.5 * w_run * DT * casadi.sumsqr(var_as[t])            # Control effort

        if use_pose_error:
            final_error = self.error6_tool(var_xs[T][:self.model.nq], param_target_pos, param_target_rot)
        else:
            final_error = self.error3_tool(var_xs[T][:self.model.nq], param_target_pos)
        totalcost += w_term * casadi.sumsqr(final_error)

        opti.subject_to(var_xs[0] == param_x0)

        for t in range(T):
            # Add dynamics constraints (simplified for demo)
            opti.subject_to(var_xs[t + 1] == var_xs[t] + DT * casadi.vertcat(var_xs[t][self.model.nq:], var_as[t]))

        for t in range(T + 1):
            for j in range(self.model.nq):
                opti.subject_to(var_xs[t][j] <= self.joint_limits['upper'][j])
                opti.subject_to(var_xs[t][j] >= self.joint_limits['lower'][j])
                if t < T:
                    opti.subject_to(casadi.fabs(var_xs[t][self.model.nq + j]) <= self.velocity_limits[j])

        opti.minimize(totalcost)
        opti.solver("ipopt", {"print_level": 0, "max_iter": 100})

        return {
            "opti": opti,
            "param_x0": param_x0,
            "param_target_pos": param_target_pos,
            "param_target_rot": param_target_rot,
            "var_xs": var_xs,
            "var_as": var_as,
        }
    def solve_mpc(self, mpc_problem, initial_q, initial_v, target_pos, target_rot, mpc_steps=10):
        """
        Solve the MPC problem using the pre-defined optimization problem.
        
        Args:
            mpc_problem (dict): Pre-defined MPC problem from `define_mpc`
            initial_q (np.ndarray): Initial joint configuration
            initial_v (np.ndarray): Initial joint velocities
            target_pos (np.ndarray): Target position
            target_rot (np.ndarray): Target rotation matrix
            mpc_steps (int): Number of MPC steps to execute
            
        Returns:
            list: Executed joint trajectories
        """
        # Extract elements from the MPC problem
        opti = mpc_problem["opti"]
        param_x0 = mpc_problem["param_x0"]
        param_target_pos = mpc_problem["param_target_pos"]
        param_target_rot = mpc_problem["param_target_rot"]
        var_xs = mpc_problem["var_xs"]
        var_as = mpc_problem["var_as"]

        # Initialize trajectory
        executed_trajectory = []
        current_q = initial_q
        current_v = initial_v

        # MPC loop
        for step in range(mpc_steps):
            # Update parameters with the current state and target
            opti.set_value(param_x0, np.hstack([current_q, current_v]))
            opti.set_value(param_target_pos, target_pos)
            opti.set_value(param_target_rot, target_rot)

            # Solve the optimization problem
            try:
                sol = opti.solve()
                next_q = sol.value(var_xs[1][:self.model.nq])
                next_v = sol.value(var_xs[1][self.model.nq:])
                control_a = sol.value(var_as[0])
            except Exception as e:
                print(f"MPC failed at step {step}: {str(e)}")
                break

            # Store the executed trajectory
            executed_trajectory.append(current_q)

            # Update the current state for the next MPC step
            current_q = next_q
            current_v = next_v

        return executed_trajectory


if __name__ == "__main__":
    # Example usage
    model_path = "urdf/model.urdf"
    robot = RobotKinematics(model_path, "eef")
    
    # Define target pose
    target_pos = np.array([0.35, 0, 1.15])
    target_rot = pin.utils.rotate('y', 3)  # Example rotation around Y-axis
    
    # Initial joint configuration and velocity
    initial_q = np.zeros(robot.model.nq)
    initial_v = np.zeros(robot.model.nv)
    
    # Define the MPC problem
    mpc_problem = robot.define_mpc(T=50, use_pose_error=True)
    
    # Solve the MPC problem
    executed_trajectory = robot.solve_mpc(
        mpc_problem, 
        initial_q, 
        initial_v, 
        target_pos, 
        target_rot, 
        mpc_steps=20
    )
    
    # Output results
    print("Final joint configuration:", executed_trajectory[-1])

