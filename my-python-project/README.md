# My Python Project

## Overview
This project implements a simulation of a 4-DOF left arm robot using the MuJoCo physics engine. The simulation includes trajectory optimization using the Crocoddyl library, allowing for efficient control of the robot's movements.

## Project Structure
```
my-python-project
├── urdf
│   ├── left_arm_4_dof.xml  # MJCF model of a 4-DOF left arm
│   └── model.urdf          # URDF model of the robot
├── mpc_setup2.py           # Main script for simulation setup and control
├── new_mpc_controller.py    # Contains the RobotKinematics class for kinematics calculations
└── README.md               # Project documentation
```

## Requirements
- Python 3.x
- MuJoCo (with appropriate license)
- Crocoddyl library
- NumPy
- Pinocchio library

## Setup Instructions
1. Install the required dependencies:
   ```bash
   pip install numpy pinocchio
   ```
   Ensure that MuJoCo and Crocoddyl are properly installed and configured in your environment.

2. Place the MuJoCo license file in the appropriate directory as per the MuJoCo installation instructions.

3. Clone this repository or download the project files to your local machine.

## Running the Simulation
To run the simulation, execute the following command in your terminal:
```bash
python mpc_setup2.py
```

This will launch the MuJoCo viewer and start the simulation of the 4-DOF left arm robot. The robot will attempt to reach the specified target pose using trajectory optimization.

## Additional Information
- The `mpc_setup2.py` file contains the main logic for loading the robot model, computing errors, and running the simulation loop.
- The `new_mpc_controller.py` file includes the `RobotKinematics` class, which handles kinematic calculations and defines the MPC problem.

For any issues or contributions, please open an issue or pull request in this repository.