import mujoco
import numpy as np
from mujoco import viewer
import time

def main():
    # 加载模型
    model_path = "urdf/left_arm_4_dof.xml"  # MJCF 文件路径
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 创建 MjvOption 对象并设置默认选项
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)  # 设置默认可视化选项

    # 创建查看器并启用 GPU 渲染
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置查看器使用 GPU
        viewer._render_context_offscreen.ctx = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        viewer._render_context_offscreen.ctx.offWidth = 1920  # 设置渲染宽度
        viewer._render_context_offscreen.ctx.offHeight = 1080  # 设置渲染高度
        viewer._render_context_offscreen.ctx.offUseGPU = True  # 启用 GPU 渲染

        # 仿真循环
        start = time.time()
        while viewer.is_running():
            # 推进仿真
            step_start = time.time()
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()

            # 控制仿真速度
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()