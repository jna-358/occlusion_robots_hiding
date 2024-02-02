import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import pickle as pkl
from architectures.linear_model import LinearModel
import datetime
from utils.rotations import get_rotation_matrix_torch

checkpoint = "trained_models/pretrained_nullspace.pth"

num_steps_max = 10000

# Start and end position
start_angles = [0.0, -0.4014, -0.7679, -0.5585, 0.0, -1.0996, 0.0]


def subsample_movement(data, dist=0.01):
    to_keep = np.zeros(data.shape[0], dtype=bool)
    i_last = 0
    to_keep[0] = True
    for i in range(1, data.shape[0]):
        if np.any(np.abs(data[i, :] - data[i_last, :]) > dist):
            to_keep[i] = True
            i_last = i
    return to_keep


def main():
    # Create the model
    model = LinearModel(7, 2, 78)

    # Load the model
    model_state_dict = torch.load(checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Load the kinematic function
    with open("data/kinematic.pkl", "rb") as f:
        kinematic_data = pkl.load(f)
        print(f"Loaded kinematic data: {kinematic_data.keys()}")
        T_base = kinematic_data["T_base"]
        T_joints = kinematic_data["T_joints"]
        T_tool = kinematic_data["T_tool"]
        T_reflector = kinematic_data["T_reflector"]
        joint_axes = kinematic_data["joint_axes"]

        # Convert to torch tensors
        T_base = torch.tensor(T_base, dtype=torch.float32)
        T_joints = [torch.tensor(T_joint, dtype=torch.float32) for T_joint in T_joints]
        T_tool = torch.tensor(T_tool, dtype=torch.float32)
        T_reflector = torch.tensor(T_reflector, dtype=torch.float32)

        def kinematic_fun(angles):
            T_res = T_base
            for i_joint in range(len(T_joints)):
                T_res = T_res @ T_joints[i_joint]
                T_res = T_res @ get_rotation_matrix_torch(
                    angles[i_joint], joint_axes[i_joint]
                )
            T_res = T_res @ T_tool
            T_res = T_res @ T_reflector
            return T_res

    # Generate some input data
    X = torch.tensor(start_angles).unsqueeze(0)

    # Get initial end effector position
    T_kinematic_initial = kinematic_fun(X[0, :])
    initial_position = T_kinematic_initial[:-1, -1]
    print(f"Initial position: {initial_position}")

    # Convert to parameter
    X_param = torch.nn.Parameter(X)

    # Optimizer
    optimizer = torch.optim.Adam([X_param], lr=1e-3)

    # Bookkeeping
    angle_history = np.empty((num_steps_max, 7))
    visibility_history = np.empty(num_steps_max)
    pos_error_history = np.empty(num_steps_max)

    # Optimize
    pbar = tqdm.tqdm(range(num_steps_max))
    for i_epoch in pbar:
        optimizer.zero_grad()

        # Forward pass
        y = model(X_param)
        visibility = y[0, -1]

        # Print first epoch
        if i_epoch == 0:
            print(f"Initial visibility: {visibility:.2f}")

        # Compute kinematics
        T_kinematic = kinematic_fun(X_param[0, :])
        target_position = T_kinematic[:-1, -1]
        loss_position = torch.sum((target_position - initial_position) ** 2)

        # Save history
        angle_history[i_epoch, :] = X_param.detach().numpy()[0]
        visibility_history[i_epoch] = visibility.detach().numpy()
        pos_error_history[i_epoch] = np.sqrt(loss_position.detach().numpy())

        # Compute loss
        loss = (1 - visibility) + loss_position * 1e4

        # Stop if visibility is good enough
        if visibility > 0.999:
            angle_history = angle_history[: i_epoch + 1, :]
            visibility_history = visibility_history[: i_epoch + 1]
            pos_error_history = pos_error_history[: i_epoch + 1]
            break

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print
        pbar.set_description(
            f"Visibility: {visibility:.2f}; Position error: {loss_position:.2e}; Loss: {loss:.2e}"
        )

    # Print final angles
    angles_final = X_param.detach().numpy()[0].tolist()
    angles_final = [round(angle, 4) for angle in angles_final]
    print(f"Final angles: {angles_final}")

    # Subsample angle history
    to_keep = subsample_movement(angle_history, dist=0.01)
    angle_history = angle_history[to_keep, :]
    visibility_history = visibility_history[to_keep]
    pos_error_history = pos_error_history[to_keep]

    # Save to file
    output_dir = "data/trajectory_optimization"
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%b%d_%H-%M")
    with open(
        output_path := os.path.join(output_dir, f"trajectory_{time_str}.pkl"), "wb"
    ) as f:
        pkl.dump(
            {
                "angles": angle_history,
                "visibility": visibility_history,
            },
            f,
        )
    full_output_path = os.path.abspath(output_path).replace("\\", "/")
    print(f"Saved trajectory to {full_output_path}")

    # Plot history
    plt.figure(figsize=(8, 4))
    for i_joint in range(7):
        plt.plot(angle_history[:, i_joint], label=f"Joint {i_joint}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Angle")
    plt.title("Joint angles")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_angles.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(visibility_history, label="Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Visibility")
    plt.legend()
    plt.title("Visibility")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_visibility.png"))
    print(f"Saved plot to {png_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(pos_error_history)
    plt.xlabel("Iteration")
    plt.ylabel("Position error / m")
    plt.title("Position error")
    plt.savefig(png_path := os.path.join(output_dir, f"trajectory_pos_error.png"))
    print(f"Saved plot to {png_path}")
    plt.show()

if __name__ == "__main__":
    main()
