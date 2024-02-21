from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import tqdm
import pickle as pkl
from architectures.linear_model import LinearModel
from utils.rotations import get_rotation_matrix_torch
from time import perf_counter

checkpoint = "data/trained_models/pretrained_nullspace.pth"
device = "cuda:0"
num_steps_max = 10000

app = FastAPI()

# Define a list of allowed origins for CORS
allowed_origins = [
    "http://192.168.1.139:8000",
]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows specified origins (use ["*"] for all origins)
    allow_credentials=True,  # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load the model (prediction)
model_predict = LinearModel(7, 2, 78)
model_state_dict = torch.load(checkpoint)
model_predict.load_state_dict(model_state_dict)
model_predict.eval()
print("Model loaded (prediction)")

# Load the model (optimization)
model_optimize = LinearModel(7, 2, 78)
model_state_dict = torch.load(checkpoint)
model_optimize.load_state_dict(model_state_dict)
model_optimize.eval()
print("Model loaded (optimization)")

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


@app.get("/predict")
def read_root(
    v0: float = 0.0,
    v1: float = 0.0,
    v2: float = 0.0,
    v3: float = 0.0,
    v4: float = 0.0,
    v5: float = 0.0,
    v6: float = 0.0,
):
    start = perf_counter()

    # Parse the angles
    angles_np = np.array([v0, v1, v2, v3, v4, v5, v6])
    print(angles_np)

    test_data = torch.from_numpy(angles_np.astype(np.float32)[None, :])

    # Forward pass
    with torch.no_grad():
        test_pred = model_predict(test_data)
        test_pred = test_pred[0, -1].item()
    end = perf_counter()

    print(f"Prediction: {test_pred:.2f} in {1e3*(end - start):.2f} ms")

    return {"angles": angles_np.tolist(), "visibility": test_pred, "time": end - start}


@app.get("/optimize")
def optimize(
    v0: float = 0.0,
    v1: float = 0.0,
    v2: float = 0.0,
    v3: float = 0.0,
    v4: float = 0.0,
    v5: float = 0.0,
    v6: float = 0.0,
):
    # Get the initial end effector position
    start_angles = [v0, v1, v2, v3, v4, v5, v6]
    print(f"Start angles: {start_angles}")
    X = torch.tensor(start_angles).unsqueeze(0)
    x_initial = kinematic_fun(X[0, :])[:-1, -1]
    print(f"Initial position: {x_initial}")

    # Target visibility threshold
    target_visibility = 0.94

    # Check initial visibility
    y = model_predict(X)
    visibility = y[0, -1].item()
    initial_visibility = visibility
    print(f"Initial visibility: {visibility:.2f}")
    if visibility > target_visibility:
        return {
            "initial_position": x_initial.tolist(),
            "initial_visibility": visibility,
            "final_visibility": visibility,
            "angles": start_angles,
            "status": "already visible",
        }
    else:
        # Try to optimize
        X_param = torch.nn.Parameter(X)
        optimizer = torch.optim.Adam([X_param], lr=1e-3)
        pbar = tqdm.tqdm(range(num_steps_max))
        for i_epoch in pbar:
            optimizer.zero_grad()

            # Forward pass
            y = model_optimize(X_param)
            visibility = y[0, -1]

            # Stop if visibility is good enough
            if visibility > target_visibility:
                break

            # Compute kinematics
            T_kinematic = kinematic_fun(X_param[0, :])
            target_position = T_kinematic[:-1, -1]
            loss_position = torch.sum((target_position - x_initial) ** 2)

            # Compute loss
            loss = (1 - visibility) + loss_position * 1e4

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Print
            pbar.set_description(
                f"Visibility: {visibility:.2f}; Position error: {loss_position:.2e}; Loss: {loss:.2e}"
            )

        # Check final visibility
        y = model_predict(X_param)
        visibility = y[0, -1].item()

        if visibility > target_visibility:
            return {
                "initial_position": x_initial.tolist(),
                "initial_visibility": initial_visibility,
                "final_visibility": visibility,
                "angles": X_param.detach().numpy()[0].tolist(),
                "status": "optimized",
            }
        else:
            return {
                "initial_position": x_initial.tolist(),
                "initial_visibility": initial_visibility,
                "final_visibility": visibility,
                "angles": None,
                "status": "no convergence",
            }


@app.get("/move")
def move(
    a0: float = 0.0,
    a1: float = 0.0,
    a2: float = 0.0,
    a3: float = 0.0,
    a4: float = 0.0,
    a5: float = 0.0,
    a6: float = 0.0,
    v0: float = 0.0,
    v1: float = 0.0,
    v2: float = 0.0,
    vis: bool = False,
):

    # Get the initial end effector position
    start_angles = [a0, a1, a2, a3, a4, a5, a6]
    print(f"Start angles: {start_angles}")
    X = torch.tensor(start_angles).unsqueeze(0)
    x_initial = kinematic_fun(X[0, :])[:-1, -1]
    print(f"Initial position: {x_initial}")

    # Target position
    x_target = torch.tensor([v0, v1, v2])
    print(f"Target position: {x_target}")

    # Try to optimize
    X_param = torch.nn.Parameter(X)
    optimizer = torch.optim.Adam([X_param], lr=1e-2)

    pbar = tqdm.tqdm(range(10))
    for i_epoch in pbar:
        optimizer.zero_grad()

        # Compute kinematics
        T_kinematic = kinematic_fun(X_param[0, :])
        target_position = T_kinematic[:-1, -1]
        loss_position = torch.sum((target_position - x_target) ** 2)

        # Compute visibility
        if vis:
            y = model_predict(X_param)
            visibility = y[0, -1]
            loss_visibility = 1 - visibility
        else:
            loss_visibility = torch.tensor(0.0)

        # Compute loss
        loss = loss_visibility + loss_position * 1e1

        # Stop if position is good enough
        if loss_position < 1e-7:
            break

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print
        pbar.set_description(f"Position error: {loss_position:.2e}; Loss: {loss:.2e}")

    # # Compute kinematics
    # T_kinematic = kinematic_fun(X_param[0, :])
    # target_position = T_kinematic[:-1, -1]
    # loss_position = torch.sum((target_position - x_target) ** 2)

    # # Compute gradient
    # loss_position.backward()

    # # Read gradient from parameter
    # gradient = X_param.grad

    # # Update parameter
    # X_param = X_param - 1e-1 * gradient

    # Check final position
    visibility = 1 - loss_visibility.item()
    T_kinematic = kinematic_fun(X_param[0, :])
    print(f"Final position: {T_kinematic[:-1, -1]}")
    print(f"Final angles (rad): {X_param.detach().numpy()[0]}")
    print(f"Final angles (deg): {np.rad2deg(X_param.detach().numpy()[0])}")
    print(f"Final visibility: {visibility}")

    return {
        "initial_position": x_initial.tolist(),
        "target_position": x_target.tolist(),
        "final_position": T_kinematic[:-1, -1].tolist(),
        "angles": X_param.detach().numpy()[0].tolist(),
        "visibility": visibility if vis else 0.0,
    }


@app.get("/grad")
def grad(
    a0: float = 0.0,
    a1: float = 0.0,
    a2: float = 0.0,
    a3: float = 0.0,
    a4: float = 0.0,
    a5: float = 0.0,
    a6: float = 0.0,
    v0: float = 0.0,
    v1: float = 0.0,
    v2: float = 0.0,
):

    # Get the initial end effector position
    start_angles = [a0, a1, a2, a3, a4, a5, a6]
    print(f"Start angles: {start_angles}")
    X = torch.tensor(start_angles).unsqueeze(0)
    x_initial = kinematic_fun(X[0, :])[:-1, -1]
    print(f"Initial position: {x_initial}")

    # Target position
    x_target = torch.tensor([v0, v1, v2])
    print(f"Target position: {x_target}")

    # Try to optimize
    X_param = torch.nn.Parameter(X)

    # Compute kinematics
    T_kinematic = kinematic_fun(X_param[0, :])
    target_position = T_kinematic[:-1, -1]
    loss_position = torch.sum((target_position - x_target) ** 2)

    # Compute gradient
    loss_position.backward()

    # Read gradient from parameter
    gradient = X_param.grad

    # Update parameter
    X_param = X_param - 1e-1 * gradient

    # Check final position
    T_kinematic = kinematic_fun(X_param[0, :])
    print(f"Final position: {T_kinematic[:-1, -1]}")
    print(f"Final angles (rad): {X_param.detach().numpy()[0]}")
    print(f"Final angles (deg): {np.rad2deg(X_param.detach().numpy()[0])}")

    return {
        "initial_position": x_initial.tolist(),
        "target_position": x_target.tolist(),
        "final_position": T_kinematic[:-1, -1].tolist(),
        "angles": X_param.detach().numpy()[0].tolist(),
    }
