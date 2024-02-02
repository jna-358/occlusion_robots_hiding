import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle as pkl
from architectures.linear_model import LinearModel
import matplotlib.pyplot as plt
from utils.data import load_data
import glob
import json
from utils.random_names import generate_random_str
import ray
from ray.tune.search.optuna import OptunaSearch
import time
import argparse
import matplotlib.pyplot as plt
import scipy.optimize
from trajectory_optimization import main as trajectory_optimization_main
from train import train
import multiprocessing

def main_increasing_duration(skip=False):
    dirname = "data/Nov08_11-05-20"  # "data/real_data"
    if not skip:
        files = glob.glob(os.path.join(dirname, "*.pkl"))

        # Remove non_distance files
        files = [f for f in files if "distance" in f]

        # Remove non-subsampled files
        files = [f for f in files if "ss" in f]

        files = sorted(files)[::-1]

        # Print all files
        print("Files:")
        for f in files:
            print(f" - {f}")

        # Define results dictionary
        results = {}

        # Load base config
        with open("training_configs/increasing_durations.json", "r") as f:
            config = json.load(f)

        for file in files:
            config["FILENAME"] = file
            print(f"Training on {file}")
            _, min_bal_acc = train(config_default=config)
            print(f"Balanced accuracy returned: {min_bal_acc:.2f}")

            results[file] = min_bal_acc

        # Save results
        with open(output_path := os.path.join(dirname, "results.pkl"), "wb") as f:
            pkl.dump(results, f)
        print(f"Saved results to {output_path}")

        with open(output_path := os.path.join(dirname, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {output_path}")
    else:
        # Load results from output file
        filename = "data/Nov08_11-05-20/results.pkl"
        with open(filename, "rb") as f:
            results = pkl.load(f)
        print(f"Loaded results from {filename}")
        print(results)

    # Convert filepaths to durations
    durations = {}
    for key in results:
        filename = os.path.basename(key)
        duration = filename.split("_")[1].replace("h", "")
        duration = int(duration)
        if duration not in durations:
            durations[duration] = []
        # Invert score
        durations[duration].append(1.0 - results[key])
    print(f"Durations: {durations}")
    iterations = [len(durations[key]) for key in durations]
    for iteration in iterations:
        if iteration != iterations[0]:
            raise Exception("Iterations do not match!")
    iterations = iterations[0]
    print(f"Iterations: {iterations}")

    # Transform measurements to numpy array
    b_acc_arr = np.zeros((len(durations), iterations))
    durations_x = np.array(sorted(list(durations.keys())))
    for i, duration in enumerate(durations_x):
        b_acc_arr[i, :] = durations[duration]

    # Compute mean and std
    b_acc_mean = np.mean(b_acc_arr, axis=1)
    b_acc_std = np.std(b_acc_arr, axis=1)

    # Fit f(x) = a - b * exp(-c * x)
    def fit_fun(x, a, b, c):
        return a - b * np.exp(-c * x)

    hours_flattened = durations_x.repeat(iterations)
    vals_flattened = b_acc_arr.flatten()

    popt, pcov = scipy.optimize.curve_fit(
        fit_fun, hours_flattened, vals_flattened, p0=[1, 1, 1]
    )
    a, b, c = popt
    hours_fit = np.linspace(min(hours_flattened), max(hours_flattened), 100)
    means_fit = fit_fun(hours_fit, a, b, c)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        durations_x,
        b_acc_mean,
        yerr=b_acc_std,
        fmt="o",
        capsize=5,
        label="Balanced accuracy",
    )
    plt.plot(hours_fit, means_fit, "k--", linewidth=1, alpha=0.5)
    plt.xlabel("$T [h]$")
    plt.ylabel("bAcc")

    for filetype in ["svg", "png", "pdf"]:
        plt.savefig(
            fig_path := os.path.join(
                dirname, f"increasing_duration_vs_balanced_acc.{filetype}"
            )
        )
        print(f"Saved figure to {fig_path}")


def train_single(config_path):
    # Load config from file
    print("Parsing config...")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))

    # Run the training
    res = train(config_default=config)


def main_latency_analysis():
    device = "cpu"
    model = LinearModel(
        7,
        2,
        78,
    )

    # Load model weights
    model.load_state_dict(torch.load("trained_models/synthetic_trajectory.pth"))
    model = model.to(device)
    model.eval()

    num_test = int(1e3)
    times_backprop = np.empty((num_test,))
    times_forward = np.empty((num_test,))

    # Backward pass
    test_data = torch.randn((1, 7)).to(device)
    for i in tqdm(range(num_test)):
        time_start = time.perf_counter()
        test_pred = model(test_data)
        loss = torch.mean(test_pred)
        loss.backward()
        time_end = time.perf_counter()
        times_backprop[i] = time_end - time_start

    # Forward pass
    with torch.no_grad():
        test_data = torch.rand((1, 7)).to(device)
        for i in tqdm(range(num_test)):
            time_start = time.perf_counter()
            test_pred = model(test_data)
            # loss = torch.mean(test_pred)
            # loss.backward()
            time_end = time.perf_counter()
            times_forward[i] = time_end - time_start

    # Remove "warmup" time
    num_warmup = int(0.1 * num_test)
    times_forward = times_forward[num_warmup:]
    times_backprop = times_backprop[num_warmup:]

    # Find mean, 5% low and 1% low
    times_mean_f = np.mean(times_forward)
    times_5p_f = np.percentile(times_forward, 95)
    times_1p_f = np.percentile(times_forward, 99)
    times_max_f = np.max(times_forward)

    times_mean_b = np.mean(times_backprop)
    times_5p_b = np.percentile(times_backprop, 95)
    times_1p_b = np.percentile(times_backprop, 99)
    times_max_b = np.max(times_backprop)

    print("Forward pass:")
    print(f"Mean: {times_mean_f*1e6:.2f} us")
    print(f"5% low: {times_5p_f*1e6:.2f} us")
    print(f"1% low: {times_1p_f*1e6:.2f} us")
    print(f"Max: {times_max_f*1e6:.2f} us")
    print()
    print("Backward pass:")
    print(f"Mean: {times_mean_b*1e6:.2f} us")
    print(f"5% low: {times_5p_b*1e6:.2f} us")
    print(f"1% low: {times_1p_b*1e6:.2f} us")
    print(f"Max: {times_max_b*1e6:.2f} us")


def main_hyperopt():
    # Define default config
    config_default_filename = "training_configs/hyper_search.json"
    with open(config_default_filename, "r") as f:
        config_default = json.load(f)

    # Prepend the local path
    config_default["FILENAME"] = os.path.join(
        os.path.dirname(__file__), config_default["FILENAME"]
    )
    config_default["BASELINE_DATA"] = os.path.join(
        os.path.dirname(__file__), config_default["BASELINE_DATA"]
    )

    # Tuning config
    config = {
        "LEARNING_RATE": ray.tune.loguniform(1e-4, 1e-1),
        "NUM_HIDDEN": ray.tune.randint(50, 100),
        "HIDDEN_LAYERS": ray.tune.randint(1, 4),
    }

    # Define scheduler
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=100, grace_period=5, reduction_factor=2
    )

    # Define the tuner
    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            ray.tune.with_parameters(train, config_default=config_default),
            resources={"cpu": multiprocessing.cpu_count(), "gpu": 1},
        ),
        tune_config=ray.tune.TuneConfig(
            metric="min_bal_acc_val",
            mode="min",
            scheduler=scheduler,
            num_samples=50,
            search_alg=OptunaSearch(),
        ),
        param_space=config,
    )

    # Run the hyperparameter search
    results = tuner.fit()
    best_result = results.get_best_result("min_bal_acc_val", "min")

    print(f"Best trial config: {best_result.config}")

    # Save the best parameter combination
    time_str = datetime.now().strftime("%b%d_%H-%M")
    hyperopt_dir = "hyperopt_runs"
    os.makedirs(hyperopt_dir, exist_ok=True)
    with open(
        res_path := os.path.join(hyperopt_dir, f"hyperopt_{time_str}.json"), "w"
    ) as f:
        hyperopt_res_dict = {
            "metrics": best_result.metrics,
            "config": best_result.config,
            "config_default": config_default,
        }
        json.dump(hyperopt_res_dict, f, indent=4)
    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    choices = [
        "baseline",
        "synthetic",
        "real",
        "durations",
        "latency",
        "durations",
        "hyperparameters",
        "nullspace",
    ]
    parser.add_argument(
        "mode", type=str, help="selects the experiment mode", choices=choices
    )
    args = parser.parse_args()

    match args.mode:
        case "baseline":
            train_single(
                "training_configs/synthetic_baseline.json"
            )
        case "synthetic":
            train_single(
                "training_configs/synthetic_trajectory.json"
            )
        case "real":
            train_single(
                "training_configs/real_world_trajectory.json"
            )
        case "durations":
            main_increasing_duration(
                skip=False
            )
        case "latency":
            main_latency_analysis()
        case "hyperparameters":
            main_hyperopt()
        case "nullspace":
            trajectory_optimization_main()
        case _:
            raise NotImplementedError()
