import pinocchio as pin
import numpy as np
from pinocchio import casadi as cpin
import casadi

class RobotKinematics:
    def __init__(self, model_path, end_frame_name,predict_horzion = 100):
        """
        Initialize the robot kinematics class.
        
        Args:
            model_path (str): Path to the URDF model file
            end_frame_name (str): Name of the end effector frame
        """
        # Load URDF model
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        
        # Store end effector frame info
        self.end_frame_name = end_frame_name
        self.endEffector_ID = self.model.getFrameId(end_frame_name)
        
        # Initialize Casadi model
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        
        # Define symbolic variables
        self.cq = casadi.SX.sym("x", self.model.nq, 1)  # joint angles
        self.ceef_pos = casadi.SX.sym("eef_pos", 3, 1)  # target position
        self.ceef_rot = casadi.SX.sym("eef_rot", 3, 3)  # target rotation matrix
        
        # Update kinematics
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # Create error functions
        self._create_error_functions()

        self.joint_limits = {

            'lower': np.array([-1.2,-0.15,0,0]),
            'upper': np.array([2.3 , 1,1,2]),

        }

    def _create_error_functions(self):
        """Create the 3D and 6D error functions using Casadi"""
        # 3D position error function
        self.error3_tool = casadi.Function(
            "etool3",
            [self.cq, self.ceef_pos],
            [self.cdata.oMf[self.endEffector_ID].translation - self.ceef_pos]
        )
        
        # 6D pose error function
        self.error6_tool = casadi.Function(
            "etool6",
            [self.cq, self.ceef_pos, self.ceef_rot],
            [cpin.log6(self.cdata.oMf[self.endEffector_ID].inverse() * 
             cpin.SE3(self.ceef_rot, self.ceef_pos)).vector]
        )

    def compute_position_error(self, q, target_pos):
        """
        Compute 3D position error.
        
        Args:
            q (np.ndarray): Joint angles
            target_pos (np.ndarray): Target position
            
        Returns:
            np.ndarray: Position error
        """
        return self.error3_tool(q, target_pos)

    def compute_pose_error(self, q, target_pos, target_rot):
        """
        Compute 6D pose error.
        
        Args:
            q (np.ndarray): Joint angles
            target_pos (np.ndarray): Target position
            target_rot (np.ndarray): Target rotation matrix
            
        Returns:
            np.ndarray: Pose error
        """
        return self.error6_tool(q, target_pos, target_rot)

    def solve_trajectory_optimization(self, initial_q, target_pos, target_rot, T=1000, w_run=0.1, w_term=10.0):
        """
        Solve trajectory optimization problem.
        
        Args:
            initial_q (np.ndarray): Initial joint configuration
            target_pos (np.ndarray): Target position
            target_rot (np.ndarray): Target rotation matrix
            T (int): Number of timesteps
            w_run (float): Running cost weight
            w_term (float): Terminal cost weight
            
        Returns:
            list: Optimized joint trajectories
        """
        opti = casadi.Opti()
        var_qs = [opti.variable(self.model.nq) for t in range(T + 1)]
        
        # Define cost function
        totalcost = 0
        for t in range(T):
            totalcost += w_run * casadi.sumsqr(var_qs[t] - var_qs[t + 1])
        totalcost += w_term * casadi.sumsqr(self.error3_tool(var_qs[T], target_pos))
        
        # Add constraints
        opti.subject_to(var_qs[0] == initial_q)
        for t in range(T + 1):
            for j in range(self.model.nq):
                # limit_range = self.joint_limits['upper'][j] - self.joint_limits['lower'][j]
                # margin = limit_range * joint_limit_margin
                
                # Apply joint position limits with margin
                opti.subject_to(var_qs[t][j] <= self.joint_limits['upper'][j] )
                opti.subject_to(var_qs[t][j] >= self.joint_limits['lower'][j] )
        
        # Set up and solve optimization
        opti.minimize(totalcost)
        opti.solver("ipopt")
        
        try:
            sol = opti.solve_limited()
            sol_qs = [opti.value(var_q) for var_q in var_qs]
        except:
            print("ERROR in convergence, using debug values.")
            sol_qs = [opti.debug.value(var_q) for var_q in var_qs]
            
        return sol_qs

if __name__ == "__main__":
    # Example usage
    model_path = "urdf/model.urdf"
    robot = RobotKinematics(model_path, "eef")
    
    # Create example target pose
    target_pos = np.array([0.35 ,0 ,1.15])
    target_rot = pin.utils.rotate('y', 3)
    target_SE3 = pin.SE3(target_rot, target_pos)
    
    # Example joint configuration
    q = np.zeros(robot.model.nq)
    
    # Compute errors
    pos_error = robot.compute_position_error(q, target_pos)
    pose_error = robot.compute_pose_error(q, target_pos, target_rot)
    
    print("Position error:", pos_error)
    print("Pose error:", pose_error)
    
    # Solve trajectory optimization
    sol_qs = robot.solve_trajectory_optimization(q, target_pos, target_rot)

    print(sol_qs[-1])