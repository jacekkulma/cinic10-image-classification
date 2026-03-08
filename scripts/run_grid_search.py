import subprocess
import itertools
import sys
import os
from datetime import datetime

def run_experiment(model, opt, bs, epochs, workers):
    """Executes a single training run via main.py"""
    print(f"\n{'='*60}")
    print(f"RUNNING: Model={model.upper()} | Opt={opt.upper()} | BS={bs}")
    print(f"{'='*60}")

    # Use sys.executable to ensure we use the current virtual env's python
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--optimizer", opt,
        "--batch_size", str(bs),
        "--epochs", str(epochs),
        "--num_workers", str(workers)
    ]

    try:
        # check=True will raise an error if main.py crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Experiment Failed: {model} with {opt}. Error: {e}")

def main():
    # 1. Define your Grid
    models = ["resnet18", "efficientnet_b0"]
    optimizers = ["adamw", "sgd"]
    batch_sizes = [128] # You can add 32 or 64 here later
    
    # Static settings
    epochs = 5
    num_workers = 4 # Optimized for your WSL/Laptop setup
    
    # 2. Generate all combinations using Cartesian Product
    experiments = list(itertools.product(models, optimizers, batch_sizes))
    
    total = len(experiments)
    print(f"Starting Grid Search: {total} experiments queued.")
    start_time = datetime.now()

    # 3. Loop through and run
    for i, (model, opt, bs) in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{total}")
        run_experiment(model, opt, bs, epochs, num_workers)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE in {duration}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()