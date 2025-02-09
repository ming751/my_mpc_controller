import mujoco
import numpy as np
from mujoco import viewer
import time
import os
import pinocchio as pin



def main():
    # 加载模型
    model_path = "urdf/left_arm_4_dof.xml"  # MJCF文件路径
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data)

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'visual_test')
    end_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'eef_site')

    from new_mpc_controller import RobotKinematics

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
    mpc_prob = robot.define_mpc(T=10, w_run=0.00001, run_error = 0.1, w_term=100.0, DT=0.05)
    i = 0
    sol_qs = np.zeros(4)
    print(len(sol_qs))

    incidator = 0
    start_time = time.time()
    while True:
        q_current = data.qpos[:]
        v_current = data.qvel[:]
        
        # 增加重规划频率，mpc控制频率为20hz
        if i % 50 == 0:
            state_traj, control_traj = robot.solve_mpc(
                mpc_problem=mpc_prob,
                initial_q=q_current,
                initial_v=v_current,
                target_pos=target_pos,
                target_rot=target_rot
            )
            
            # 使用第一个控制输入
            #if robot.last_successful_solve:
            #data.ctrl[:4] = state_traj[1][:4]

            data.ctrl[:4] = state_traj[1][:4]
            # else:
            #     # 可以添加额外的保护措施，比如减小控制信号
            #     data.ctrl[:4] = control_traj[0] * 0.5
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # 添加目标更新逻辑的保护
        current_pos = data.site_xpos[end_site_id][:3]
        error = np.linalg.norm(current_pos - target_pos)
        print(f"wall_time: {time.time() - start_time}, 当前位置: {current_pos}, 目标位置: {target_pos}, 误差: {error}")
        
        if error < 0.005:
            if not incidator:  # 避免重复打印
                print(f"到达目标点！当前误差: {error}")
                incidator = True
        
        if incidator:
            # 限制目标移动速度
            target_pos = target_pos + np.array([0.00, -0.001, -0.001])
            # 可选：添加目标位置的边界检查
            
        i += 1
    

            

if __name__ == "__main__":
    main()