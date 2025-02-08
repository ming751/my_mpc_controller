import pinocchio as pin
import numpy as np

# 加载 URDF 模型
model_path = "urdf/left_arm_4_dof.xml"
model = pin.buildModelFromMJCF(model_path)
data = model.createData()

# 设置关节配置 (这里使用示例值，你可以根据需要修改)
q = np.array([0.5, -0.3, 0.8, 0.2])  
print('关节角度配置:', q)

# 计算前向运动学
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)  # 更新所有frame的位置

# 获取末端执行器的帧 ID
frame_id = model.getFrameId("eef")

# 获取末端执行器相对于世界坐标系的位姿
eef_placement = data.oMf[frame_id]  # oMf 表示从世界坐标系到frame的变换

# 1. 位置信息 (3x1 向量)
position = eef_placement.translation
print("\n末端执行器位置:")
print(f"X: {position[0]:.4f}")
print(f"Y: {position[1]:.4f}")
print(f"Z: {position[2]:.4f}")

# 2. 姿态信息
# 2.1 旋转矩阵 (3x3)
rotation_matrix = eef_placement.rotation
print("\n旋转矩阵:")
print(rotation_matrix)

# 2.2 转换为RPY角 (roll-pitch-yaw)
rpy = pin.rpy.matrixToRpy(rotation_matrix)
print("\nRPY角 (弧度):")
print(f"Roll: {rpy[0]:.4f}")
print(f"Pitch: {rpy[1]:.4f}")
print(f"Yaw: {rpy[2]:.4f}")

# 2.3 转换为四元数
quat = pin.Quaternion(rotation_matrix)
print("\n四元数 [x, y, z, w]:")


# 3. 齐次变换矩阵 (4x4)
homogeneous_matrix = eef_placement.homogeneous
print("\n齐次变换矩阵:")
print(homogeneous_matrix)

# # 4. 验证运动链的完整性
# print("\n运动链中的关节位置:")
# for i, frame in enumerate(model.frames):
#     if frame.type != pin.FrameType.FIXED_JOINT:
#         print(f"\n{frame.name}:")
#         print(f"Position: {data.oMf[i].translation}")
#         print(f"Rotation:\n{data.oMf[i].rotation}")