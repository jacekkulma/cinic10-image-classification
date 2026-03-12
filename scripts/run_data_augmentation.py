import subprocess
import itertools
import sys
import os
from datetime import datetime

def run_experiment(model, opt, augmentation_type, bs, epochs, workers, dropout, weight_decay):
    """Executes a single training run via main.py with augmentation configuration"""
    print(f"\n{'='*60}")
    print(f"RUNNING: Model={model.upper()} | Opt={opt.upper()}")
    print(f"Augmentation Type={augmentation_type.upper()}")
    print(f"{'='*60}")

    # Use sys.executable to ensure we use the current virtual env's python
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--optimizer", opt,
        "--batch_size", str(bs),
        "--epochs", str(epochs),
        "--num_workers", str(workers),
        "--seed", "42",  # Fixed seed for reproducibility
        "--dropout", str(dropout),
        "--weight_decay", str(weight_decay),
        "--augmentation_type", augmentation_type,
    ]

    try:
        # check=True will raise an error if main.py crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Experiment Failed: {model} with {opt} and augmentation_type={augmentation_type}. Error: {e}")

def main():
    # 1. Define your Grid
    models = ["vgg16", "resnet18", "efficientnet_b0"]    
    augmentation_types = ["none", "simple", "advanced", "both"]

    # Static settings
    batch_size = 32
    optimizer = "adamw"
    epochs = 1
    num_workers = 0  # Set according to your device setup
    dropout_rate = 0.1
    weight_decay = 1e-3
    
    # 2. Generate all combinations using Cartesian Product
    experiments = list(itertools.product(models, augmentation_types))
    
    total = len(experiments)
    print(f"Starting Data Augmentation Experiment: {total} experiments queued.")
    start_time = datetime.now()

    # 3. Loop through and run
    for i, (model, aug_type) in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{total}")
        run_experiment(model, optimizer, aug_type, batch_size, epochs, num_workers, dropout_rate, weight_decay)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"DATA AUGMENTATION EXPERIMENTS COMPLETE in {duration}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
