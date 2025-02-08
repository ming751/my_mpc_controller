import pinocchio as pin
import numpy as np
from pinocchio import casadi as cpin
import casadi

# 加载 URDF 模型
model_path = "urdf/model.urdf"
model = pin.buildModelFromUrdf(model_path)
data = model.createData()

# 初始配置
q_neural = np.zeros(model.nv)
end_frame_name = 'eef'
endEffector_ID = model.getFrameId(end_frame_name)

# Casadi 模型设置
cmodel = cpin.Model(model)
cdata = cmodel.createData()

# 定义符号变量
cq = casadi.SX.sym("x", model.nq, 1)  # 关节角度
ceef_pos = casadi.SX.sym("eef_pos", 3, 1)  # 目标位置
ceef_rot = casadi.SX.sym("eef_rot", 3, 3)  # 目标旋转矩阵

# 更新运动学
cpin.framesForwardKinematics(cmodel, cdata, cq)

# 3D位置误差计算函数
error3_tool = casadi.Function(
    "etool3",
    [cq, ceef_pos],  # 输入关节角度和目标位置
    [cdata.oMf[endEffector_ID].translation - ceef_pos]  # 位置误差
)

# 6D位姿误差计算函数
error6_tool = casadi.Function(
    "etool6",
    [cq, ceef_pos, ceef_rot],  # 输入关节角度、目标位置和旋转矩阵
    [cpin.log6(cdata.oMf[endEffector_ID].inverse() * cpin.SE3(ceef_rot, ceef_pos)).vector]  # 位姿误差
)

# 使用示例
if __name__ == "__main__":
    # 创建示例目标位姿
    target_pos = np.array([-0.5, 0.1, 0.2])
    target_rot = pin.utils.rotate('y', 3)
    target_SE3 = pin.SE3(target_rot, target_pos)
    
    # 计算误差示例
    q = np.zeros(model.nq)  # 示例关节角度
    
    # 计算3D位置误差
    pos_error = error3_tool(q, target_pos)
    print("Position error:", pos_error)
    
    # 计算6D位姿误差
    pose_error = error6_tool(q, target_pos, target_rot)
    print("Pose error:", pose_error)

    T = 1000
    w_run = 0.1
    w_term = 1

    opti = casadi.Opti()
    var_qs = [opti.variable(model.nq) for t in range(T + 1)]
    print(var_qs[0].shape)
    totalcost = 0
    for t in range(T):
        totalcost += w_run * casadi.sumsqr(var_qs[t] - var_qs[t + 1])
    totalcost += w_term * casadi.sumsqr(error_tool(var_qs[T]))

    opti.subject_to(var_qs[0] == q)

    # %load tp2/generated/trajopt_kine_solve
    opti.minimize(totalcost)
    opti.solver("ipopt")  # set numerical backend
    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        sol_qs = [opti.value(var_q) for var_q in var_qs]
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_qs = [opti.debug.value(var_q) for var_q in var_qs]


