import pinocchio as pin
import numpy as np
from pinocchio import casadi as cpin
import casadi
import time
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
        #print(self.endEffector_ID)
        
        # Initialize Casadi model
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        #Get nq and nv 
        nq = self.model.nq
        nv = self.model.nv

        #State dims
        nx = nq + nv

        #Dot state dims 
        ndx = 2 * nv
        #print(self.model.nq,self.model.nv)

        # Define symbolic variables
        self.cx = casadi.SX.sym("x", nx, 1)  # joint angles
        self.cdx = casadi.SX.sym("dx", nv * 2 , 1)
        self.cq = self.cx[:nq]
        self.cv = self.cx[nq:]
        self.caq = casadi.SX.sym("a",nv,1)

        self.ceef_pos = casadi.SX.sym("eef_pos", 3, 1)  # target position
        self.ceef_rot = casadi.SX.sym("eef_rot", 3, 3)  # target rotation matrix
        
        # Update kinematics
        cpin.forwardKinematics(self.cmodel, self.cdata, self.cq,self.cv,self.caq)
        cpin.updateFramePlacements(self.cmodel,self.cdata)
        
        # Create error functions
        self._create_error_functions()

        self.joint_limits = {

            'lower': np.array([-1.2,-0.15,0,0]),
            'upper': np.array([2.3 , 1,1,2]),

        }

        self.error_scalar = None

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
    
    def define_mpc(self, T=100, w_run=0.1, run_error = 0.1,w_term=10.0, DT=0.002):
        """
        Define the MPC optimization problem and return the Casadi Opti object.
        
        Args:
            T (int): Prediction horizon
            w_run (float): Running cost weight
            w_term (float): Terminal cost weight
            DT (float): Time step duration
            
        Returns:
            dict: Contains the Opti object, parameters, and variables
        """
        # Initialize optimization problem
        cnext = casadi.Function(
            "next",
            [self.cx, self.caq],
            [
                casadi.vertcat(
                    cpin.integrate(self.cmodel, self.cq, self.cv * DT + self.caq * DT**2),
                    self.cv + self.caq * DT,
                )
            ],
        )

        opti = casadi.Opti()

        # Define parameters for the initial state and target
        param_x0 = opti.parameter(self.model.nq + self.model.nv)  # Initial state
        param_target_pos = opti.parameter(3)                      # Target position
        param_target_rot = opti.parameter(3, 3)                   # Target rotation

        # Define optimization variables for state and control
        var_xs = [opti.variable(self.model.nq + self.model.nv) for t in range(T + 1)]
        var_as = [opti.variable(self.model.nv) for t in range(T)]

        # Define the cost function
        totalcost = 0
        for t in range(T):
            totalcost += w_run * DT * casadi.sumsqr(var_xs[t][self.model.nq:])  # Running cost
            totalcost += 0.5 * w_run * DT * casadi.sumsqr(var_as[t])  
            totalcost += run_error * casadi.sumsqr(self.error3_tool(var_xs[t][:self.model.nq], param_target_pos)  )        # Control effort

        final_error = self.error3_tool(var_xs[T][:self.model.nq], param_target_pos)

        totalcost += w_term * casadi.sumsqr(final_error)                        # Terminal cost

        # Initial state constraint
        opti.subject_to(var_xs[0] == param_x0)

        # Dynamic constraints
        for t in range(T):
            opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])

        # Joint limit constraints
        for t in range(T + 1):
            for j in range(self.model.nq):
                opti.subject_to(var_xs[t][j] <= self.joint_limits['upper'][j])
                opti.subject_to(var_xs[t][j] >= self.joint_limits['lower'][j])

        #Joint velocity constraints
        for t in range(T):
            for j in range(self.model.nv):
                opti.subject_to(var_xs[t][self.model.nq+j] <= 10)
                opti.subject_to(var_xs[t][self.model.nq+j] >= -10)

        # Set the solver with correct IPOPT options
        opti.minimize(totalcost)
        opti.solver('ipopt', {
            'print_time': False,
            'ipopt': {
                'print_level': 0,  # This goes inside the 'ipopt' dictionary
                'sb': 'yes'        # Suppress IPOPT banner
            }
        })
        #opti.solver("ipopt", opts)

        # Return the Opti object and variables
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
        使用预定义的 MPC 优化问题求解，并添加热启动功能和错误处理
        """
        try:
            # 提取 MPC 问题中的元素
            opti = mpc_problem["opti"]
            param_x0 = mpc_problem["param_x0"]
            param_target_pos = mpc_problem["param_target_pos"]
            param_target_rot = mpc_problem["param_target_rot"]
            var_xs = mpc_problem["var_xs"]
            var_as = mpc_problem["var_as"]

            # 设置当前状态和目标
            x0 = np.hstack([initial_q, initial_v])
            opti.set_value(param_x0, x0)
            opti.set_value(param_target_pos, target_pos)
            opti.set_value(param_target_rot, target_rot)

            # 热启动：使用上一次的解作为初始猜测
            if hasattr(self, "warm_start_x") and hasattr(self, "warm_start_u"):
                # 状态变量热启动
                for t, var_x in enumerate(var_xs[:-1]):  # 除了最后一个状态
                    next_state = self.warm_start_x[t+1] if t+1 < len(self.warm_start_x) else self.warm_start_x[-1]
                    opti.set_initial(var_x, next_state)
                # 最后一个状态使用目标状态
                opti.set_initial(var_xs[-1], self.warm_start_x[-1])
                
                # 控制变量热启动
                for t, var_a in enumerate(var_as):
                    next_control = self.warm_start_u[t] if t < len(self.warm_start_u) else self.warm_start_u[-1]
                    opti.set_initial(var_a, next_control)

            # 求解优化问题
            sol = opti.solve()

            # 提取最优轨迹
            state_traj = [sol.value(x) for x in var_xs]
            control_traj = [sol.value(u) for u in var_as]

            # 保存当前解用于下次热启动
            self.warm_start_x = state_traj
            self.warm_start_u = control_traj

            # 记录求解状态
            self.last_successful_solve = True
            return state_traj, control_traj

        except Exception as e:
            print(f"MPC求解失败: {str(e)}")
            self.last_successful_solve = False
            
            # 如果有上一次的解，使用上一次的解
            if hasattr(self, "warm_start_x") and hasattr(self, "warm_start_u"):
                print("使用上一次的解...")
                # 移除第一个状态，复制最后一个状态
                state_traj = self.warm_start_x[1:] + [self.warm_start_x[-1]]
                control_traj = self.warm_start_u[1:] + [self.warm_start_u[-1]]
                return state_traj, control_traj
            else:
                # 如果没有上一次的解，返回零控制
                print("没有可用的上一次解，返回零控制...")
                T = len(var_xs)
                state_traj = [x0] * T
                control_traj = [np.zeros(self.model.nv)] * (T-1)
                return state_traj, control_traj



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

    mpc_prob = robot.define_mpc(T=100, w_run=0.1, w_term=10.0, DT=0.002)
    start_time = time.time()
    
    # Solve trajectory optimization
    sol_qs , _ = robot.solve_mpc(mpc_problem=mpc_prob,initial_q=q, initial_v = np.zeros(4) ,target_pos=target_pos,target_rot= target_rot)
    end_time = time.time()
    print(f"MPC 求解时间: {end_time - start_time} 秒")
    print(sol_qs[-1])

    pin.forwardKinematics(robot.model, robot.data, sol_qs[-1][:robot.model.nq], np.zeros(robot.model.nv), np.zeros(robot.model.nv))
    pin.updateFramePlacements(robot.model, robot.data)
    final_pos = robot.data.oMf[robot.endEffector_ID].translation
    print("final_pos",final_pos)
    pos_error = robot.compute_position_error(sol_qs[-1][:robot.model.nq], target_pos)

    print(pos_error)