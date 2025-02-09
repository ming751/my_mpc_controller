import sys
import os
import time
try:
    import pinocchio as pin
    import numpy as np
    from pinocchio import casadi as cpin
    import casadi
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ImportError as e:
    print(f"错误: 缺少必要的依赖项 - {str(e)}")
    print("请确保已安装以下包:")
    print("- pinocchio")
    print("- numpy")
    print("- casadi")
    print("- acados_template")
    sys.exit(1)

# 检查 ACADOS_SOURCE_DIR 环境变量是否设置
if 'ACADOS_SOURCE_DIR' not in os.environ:
    print("警告: 未设置 ACADOS_SOURCE_DIR 环境变量")
    print("请设置环境变量，例如:")
    print("export ACADOS_SOURCE_DIR=/path/to/acados")
    sys.exit(1)

class RobotKinematics:
    def __init__(self, model_path, end_frame_name, predict_horizon=100):
        """
        初始化机器人运动学类。
        
        参数:
            model_path (str): URDF 模型文件路径
            end_frame_name (str): 末端执行器帧名称
            predict_horizon (int): 预测地平线步数
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到URDF模型文件: {model_path}")

        # 加载 URDF 模型
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        
        # 存储末端执行器帧信息
        self.end_frame_name = end_frame_name
        self.endEffector_ID = self.model.getFrameId(end_frame_name)
        
        # 初始化状态
        self.predict_horizon = predict_horizon

        # 定义关节限制
        self.joint_limits = {
            'lower': np.array([-1.2,-0.15,0,0]),
            'upper': np.array([2.3, 1,1,2]),
        }

        # 初始化 Casadi 模型
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # 获取维度信息
        nq = self.model.nq
        nv = self.model.nv
        nx = nq + nv
        
        # 定义符号变量
        self.cx = casadi.SX.sym("x", nx, 1)  # 状态变量 [q, v]
        self.cq = self.cx[:nq]  # 位置部分
        self.cv = self.cx[nq:]  # 速度部分
        self.cu = casadi.SX.sym("u", nv, 1)  # 控制输入

        # 定义目标位姿符号变量
        self.ceef_pos = casadi.SX.sym("eef_pos", 3, 1)
        self.ceef_rot = casadi.SX.sym("eef_rot", 3, 3)

        # 定义 acados MPC
        self.ocp = self.define_mpc()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

        # 初始化热启动变量
        self.warm_start_u = np.zeros((self.predict_horizon, nv))
        self.warm_start_x = np.zeros((self.predict_horizon + 1, nx))

        # 创建误差函数
        self._create_error_functions()

        self.error_scalar = None

    def create_acados_model(self):
        """
        创建 acados 优化模型，考虑二阶运动学
        """
        model = AcadosModel()
        model.name = 'robot_arm'

        # 获取维度
        nq = self.model.nq  # 4
        nv = self.model.nv  # 4
        nx = nq + nv        # 8
        nu = nv             # 4

        # 定义状态和控制变量
        x = casadi.SX.sym('x', nx)  # [q, v]
        u = casadi.SX.sym('u', nu)  # 角加速度
        xdot = casadi.SX.sym('xdot', nx)
        
        # 提取状态变量
        q = x[:nq]  # 关节角度 (4)
        v = x[nq:]  # 关节速度 (4)

        # 时间步长
        dt = 0.001

        # 状态更新方程
        q_next = q + v*dt + 0.5*u*dt**2  # 二阶泰勒展开
        v_next = v + u*dt                 # 一阶欧拉积分
        
        # 计算末端位置
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        current_pos = self.cdata.oMf[self.endEffector_ID].translation
        
        # 定义目标位置参数
        p = casadi.SX.sym('p', 3)  # 目标位置
        
        # 运行时代价输出 (3位置误差 + 4控制量)
        pos_error = current_pos - p
        y = casadi.vertcat(pos_error, u)  # 7维输出
        
        # 终端代价输出 (仅位置误差)
        y_e = pos_error  # 3维输出
        
        # 状态空间方程
        x_next = casadi.vertcat(q_next, v_next)
        f_expl = casadi.vertcat(v, u)  # 连续时间动力学
        
        # 配置模型
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = p
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl
        model.disc_dyn_expr = x_next
        
        # 显式设置代价函数表达式
        model.cost_y_expr = y       # 运行时代价输出
        model.cost_y_expr_e = y_e   # 终端代价输出

        return model

    def define_mpc(self):
        """
        定义 MPC 优化问题
        """
        ocp = AcadosOcp()
        
        # 基本参数
        self.dt = 0.05
        ocp.model = self.create_acados_model()
        ocp.dims.N = self.predict_horizon
        ocp.solver_options.tf = self.dt * self.predict_horizon

        # 设置参数维度
        ocp.dims.np = 3  # 目标位置参数维度 (x,y,z)
        
        # 初始化参数值
        ocp.parameter_values = np.zeros(3)

        # 维度信息
        nq = self.model.nq  # 4
        nv = self.model.nv  # 4
        nu = nv  # 4
        
        # 显式设置输出维度
        ocp.dims.ny = 3 + nu  # 7 (3位置误差 + 4控制量)
        ocp.dims.ny_e = 3     # 终端误差维度

        # 明确配置离散时间模型
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.sim_method_num_steps = 1

        # 代价函数配置
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        
        # 运行时代价权重矩阵 (7x7)
        W = np.diag([5000.0, 5000.0, 5000.0] + [0.01]*nu)
        
        # 终端代价权重矩阵 (3x3)
        W_e = 50000.0 * np.eye(3)
        
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        
        # 参考轨迹 (全部设为零，因为y已经是误差项)
        ocp.cost.yref = np.zeros(ocp.dims.ny)
        ocp.cost.yref_e = np.zeros(ocp.dims.ny_e)

        # 状态约束
        ocp.constraints.lbx = np.hstack([
            self.joint_limits['lower'],  # 关节角度下限 (4)
            -10.0 * np.ones(nv)         # 速度下限 (4)
        ])
        ocp.constraints.ubx = np.hstack([
            self.joint_limits['upper'],  # 关节角度上限 (4)
            10.0 * np.ones(nv)          # 速度上限 (4)
        ])
        ocp.constraints.idxbx = np.arange(nq + nv)  # 约束所有状态
        
        # 控制约束
        ocp.constraints.lbu = -20.0 * np.ones(nu)
        ocp.constraints.ubu = 20.0 * np.ones(nu)
        ocp.constraints.idxbu = np.arange(nu)

        # 求解器配置
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.max_iter = 200

        ocp.solver_options.N_horizon = self.predict_horizon

        return ocp

    def solve_mpc(self, mpc_problem, initial_q, initial_v, target_pos, target_rot, mpc_steps=10):
        """
        求解 MPC 问题
        """
        try:
            # 设置初始状态
            x0 = np.hstack([initial_q, initial_v])
            self.solver.set(0, "x", x0)
            
            # 设置目标位置参数
            for i in range(self.predict_horizon + 1):
                self.solver.set(i, "p", target_pos)

            # 求解
            status = self.solver.solve()

            if status == 0:
                # 提取轨迹
                state_traj = []
                control_traj = []
                for i in range(self.predict_horizon + 1):
                    state_traj.append(self.solver.get(i, "x"))
                    if i < self.predict_horizon:
                        control_traj.append(self.solver.get(i, "u"))
                
                return state_traj, control_traj
            else:
                print("acados 求解失败，状态码:", status)
                return None, None

        except Exception as e:
            print(f"MPC 求解失败: {str(e)}")
            return None, None

    def compute_position_error(self, q, target_pos):
        """
        计算3D位置误差。
        
        参数:
            q (np.ndarray): 关节角
            target_pos (np.ndarray): 目标位置
            
        返回:
            np.ndarray: 位置误差
        """
        return self.error3_tool(q, target_pos)

    def compute_pose_error(self, q, target_pos, target_rot):
        """
        计算6D姿态误差。
        
        参数:
            q (np.ndarray): 关节角
            target_pos (np.ndarray): 目标位置
            target_rot (np.ndarray): 目标旋转矩阵
            
        返回:
            np.ndarray: 姿态误差
        """
        return self.error6_tool(q, target_pos, target_rot)

    def _create_error_functions(self):
        """
        创建误差计算函数
        """
        # 更新运动学
        cpin.forwardKinematics(self.cmodel, self.cdata, self.cq)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        
        # 3D位置误差
        current_pos = self.cdata.oMf[self.endEffector_ID].translation
        self.error3_tool = casadi.Function(
            "pos_error",
            [self.cq, self.ceef_pos],
            [current_pos - self.ceef_pos]
        )
        
        # 6D姿态误差
        current_pose = self.cdata.oMf[self.endEffector_ID]
        target_pose = cpin.SE3(self.ceef_rot, self.ceef_pos)
        self.error6_tool = casadi.Function(
            "pose_error",
            [self.cq, self.ceef_pos, self.ceef_rot],
            [cpin.log6(current_pose.inverse() * target_pose).vector]
        )

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

    mpc_prob = robot.define_mpc()#T=1000, w_run=0.1, w_term=10.0, DT=0.002)
    mpc_prob = robot.define_mpc()
    
    # Solve trajectory optimization
    start_time = time.time()
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
    sol_qs, sol_us = robot.solve_mpc(
        mpc_problem=mpc_prob,
        initial_q=q,
        initial_v=np.zeros(robot.model.nv),
        target_pos=target_pos,
        target_rot=target_rot
    )

    solved_qs = sol_qs[-1] if sol_qs is not None else None

    pin.forwardKinematics(robot.model, robot.data, solved_qs[:robot.model.nq])
    pin.updateFramePlacements(robot.model, robot.data)
    
    # 3D位置误差
    solved_pos = robot.data.oMf[robot.endEffector_ID].translation

    print(solved_pos)

    print(sol_qs[-1] if sol_qs is not None else "无解")