import mujoco
import numpy as np
from mujoco import viewer
import time
import os
import pinocchio as pin
import crocoddyl



def main():
    # Load model
    model_path = "urdf/left_arm_4_dof.xml"  # MJCF file path
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data)

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'visual_test')

    joint_limits = {

        'lower': np.array([-1.2,-0.15,0,0]),
        'upper': np.array([2.3 , 1,1,2]),

    }

    from new_mpc_controller import RobotKinematics

    model_path = "urdf/model.urdf"
    pin_model= pin.buildModelFromUrdf(model_path)
    try:
        robot = RobotKinematics(model_path, "eef")
    except ValueError as e:
        print(e)
        return
    
    # Create example target pose
    target_pos = np.array([0.35 ,0 ,1.15])
    target_rot = pin.utils.rotate('y', 3)
    target_SE3 = pin.SE3(target_rot, target_pos)
    
    # Example joint configuration
    q = np.zeros(pin_model.nq)
    
    # Define Crocoddyl problem
    dt = 0.02
    T = 100
    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFull(state)
    
    # Define cost functions
    running_cost_model = crocoddyl.CostModelSum(state, actuation.nu)
    terminal_cost_model = crocoddyl.CostModelSum(state, actuation.nu)
    
    # Goal tracking cost
    end_effector_id = pin_model.getFrameId("eef")
    goal_residual = crocoddyl.ResidualModelFrameTranslation(state, end_effector_id, target_pos)
    goal_tracking_cost = crocoddyl.CostModelResidual(state, goal_residual)
    
    # State and control regularization costs
    x_reg_cost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
    u_reg_cost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))
    
    # Add costs to running and terminal cost models
    running_cost_model.addCost("goal_tracking", goal_tracking_cost, 10.0)
    running_cost_model.addCost("x_reg", x_reg_cost, 1e-3)
    
    running_cost_model.addCost("u_reg", u_reg_cost, 1e-2)
    
    terminal_cost_model.addCost("goal_tracking", goal_tracking_cost, 1.0)

    # pin_model.jointLimit = joint_limits['lower']

    # joint_pos_constraint = crocoddyl.ConstraintModelBounds(state, joint_limits['lower'], joint_limits['upper'], crocoddyl.ConstraintType.State)

    # # 创建约束管理器
    # constraints = crocoddyl.ConstraintModelManager(state, model.nv)

    # # 添加关节角度约束
    # constraints.addConstraint("joint_pos_constraint", joint_pos_constraint)

    #actuation.set_constraints(constraints)

    

    def add_joint_limits_cost(cost_model, state, joint_limits):
        lower_bound = joint_limits['lower']
        upper_bound = joint_limits['upper']
        
        for i in range(state.nq):
            lower_residual = crocoddyl.ResidualModelJointLimits(state, i, lower_bound[i], upper_bound[i])
            upper_residual = lower_residual  # Use the same residual for both lower and upper limits
            
            lower_cost = crocoddyl.CostModelResidual(state, lower_residual)
            upper_cost = crocoddyl.CostModelResidual(state, upper_residual)
            
            cost_model.addCost(f"joint_lower_limit_{i}", lower_cost, 1e2)
            cost_model.addCost(f"joint_upper_limit_{i}", upper_cost, 1e2)

    add_joint_limits_cost(running_cost_model, state, joint_limits)
    
    # Define action models
    running_model = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, running_cost_model), dt)
    terminal_model = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_cost_model), dt)
    
    problem = crocoddyl.ShootingProblem(np.zeros(state.nx), [running_model] * T, terminal_model)

    # Initialize the solver
    solver = crocoddyl.SolverFDDP(problem)

    i = 0
    sol_qs = np.zeros((T, robot.model.nq))
    
    while True:
        q_current = data.qpos[:]
        v_current = data.qvel[:]
        x_current = np.concatenate([q_current, v_current])  # Concatenate q and v to form the state vector
        if i % 20 == 0:
            solver.solve(init_xs=[x_current] * (T + 1), init_us=[np.zeros(actuation.nu)] * T)
            sol_qs = np.array(solver.xs)[:, :robot.model.nq]
        
        if i < len(sol_qs):
            data.ctrl[:4] = sol_qs[i]
        else:
            data.ctrl[:4] = sol_qs[-1]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.02)
        i += 1

if __name__ == "__main__":
    main()