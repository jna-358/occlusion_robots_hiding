import torch
import numpy as np
import pickle as pkl


def split_test_middle(X, Y, test_split=0.2, buffer_split=0.01):
    # Compute number of samples
    num_samples = X.shape[0]

    # Split into train and test set (train buffer test buffer train)
    num_test = int(test_split * num_samples)
    num_buffer_half = int(buffer_split * num_samples)
    num_train = num_samples - num_test - 2 * num_buffer_half
    num_train_half = num_train // 2

    is_train = np.zeros((num_samples,), dtype=bool)
    is_test = np.zeros((num_samples,), dtype=bool)
    is_train[:num_train_half] = True
    is_test[
        num_train_half + num_buffer_half : num_train_half + num_buffer_half + num_test
    ] = True
    is_train[num_train_half + num_buffer_half + num_test + num_buffer_half :] = True

    X_train, Y_train = X[is_train, :], Y[is_train, :]
    X_test, Y_test = X[is_test, :], Y[is_test, :]

    return X_train, Y_train, X_test, Y_test


def load_data(filename, split=None, device=None):
    # Load the data
    print(f"Loading {filename} with split={split} and device={device}")
    with open(filename, "rb") as f:
        data = pkl.load(f)

    # Read all properties
    joint_angles = data["joint_angles"].astype(np.float32)
    visibility = data["visibility"][:, 0].astype(np.float32)
    distance = data["distances"][:, 0].astype(np.float32)

    if "position" in data.keys():
        target_positions = data["position"].astype(np.float32)
    else:
        target_positions = np.zeros((joint_angles.shape[0], 3), dtype=np.float32)

    num_samples = joint_angles.shape[0]
    print(f"Loaded {num_samples} samples")

    # Concatenate all properties
    target_output = np.concatenate(
        [
            target_positions,
            np.transpose(distance[None, :]),
            np.transpose(visibility[None, :]),
        ],
        axis=1,
    )

    X = joint_angles
    Y = target_output

    # Remove NaN values
    nan_lines = np.logical_or(np.any(np.isnan(X), axis=1), np.any(np.isnan(Y), axis=1))
    X = X[np.invert(nan_lines), :]
    Y = Y[np.invert(nan_lines), :]

    if split is not None:
        # Check that split is a float
        assert isinstance(split, float), "split must be a float"

        # Split the data
        X_train, Y_train, X_test, Y_test = split_test_middle(X, Y, test_split=split)

        # Convert to torch tensors
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        Y_train = torch.from_numpy(Y_train)
        Y_test = torch.from_numpy(Y_test)

        # Shift to GPU
        if device is not None:
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
        return X_train, Y_train, X_test, Y_test
    else:
        # Convert to torch tensors
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        # Shift to GPU
        if device is not None:
            X = X.to(device)
            Y = Y.to(device)
        return X, Y
