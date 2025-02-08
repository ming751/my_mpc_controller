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


    from mpc_controller import RobotKinematics

    model_path = "urdf/model.urdf"
    robot = RobotKinematics(model_path, "eef",predict_horzion = 100)
    
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
    i = 0
    print(len(sol_qs))

    start_time = time.time()
    while True:

        q_current = data.qpos[:]
        if i % 10 == 0:
            sol_qs = robot.solve_trajectory_optimization(q_current, target_pos, target_rot,T = 2)
        # if i < len(sol_qs):
        #     data.ctrl[:4] = sol_qs[i]
        # else:
        #     data.ctrl[:4] = sol_qs[-1]

        data.ctrl[:4] = sol_qs[1]
        mujoco.mj_step(model, data)
        viewer.sync()
        wall_time_consume = time.time() - start_time
        print(wall_time_consume - data.time)
        #time.sleep(0.02)
        i += 1
    

            

if __name__ == "__main__":
    main()