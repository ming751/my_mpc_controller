import numpy as np
import crocoddyl
import pinocchio as pin

class RobotKinematics:
    def __init__(self, model_path, end_effector_name):
        self.model = pin.buildModelFromUrdf(model_path)
        self.data = self.model.createData()
        self.end_effector_id = self.model.getFrameId(end_effector_name)
        if self.end_effector_id == -1:
            raise ValueError(f"End effector '{end_effector_name}' not found in the model.")
        print(f"End effector ID: {self.end_effector_id}")

    def compute_position_error(self, q, target_pos):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        eef_pos = self.data.oMi[self.end_effector_id].translation
        return np.linalg.norm(eef_pos - target_pos)

    def compute_pose_error(self, q, target_pos, target_rot):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        eef_pose = self.data.oMi[self.end_effector_id]
        pos_error = np.linalg.norm(eef_pose.translation - target_pos)
        rot_error = np.linalg.norm(pin.log3(eef_pose.rotation.T @ target_rot))
        return pos_error + rot_error

    def define_mpc(self, T, w_run, w_term, DT):
        # Define the cost function and dynamics for the MPC problem
        running_cost = crocoddyl.CostModelState(self.model, self.data, np.zeros(self.model.nq), w_run)
        terminal_cost = crocoddyl.CostModelState(self.model, self.data, np.zeros(self.model.nq), w_term)
        
        # Create the action model
        action_model = crocoddyl.ActionModelAbstract(self.model)
        action_model.costs.addCost("running_cost", running_cost)
        action_model.costs.addCost("terminal_cost", terminal_cost)

        # Create the problem
        problem = crocoddyl.ShootingProblem(np.zeros(self.model.nq), [action_model] * T)
        return problem

    def solve_mpc(self, mpc_problem, initial_q, initial_v, target_pos, target_rot):
        # Set the initial state
        state = np.concatenate([initial_q, initial_v])
        # Create the solver
        solver = crocoddyl.SolverKKT(mpc_problem)
        # Solve the problem
        solver.solve(state, 100)
        return solver.xs  # Return the solution trajectory