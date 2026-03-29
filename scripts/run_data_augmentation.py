import subprocess
import itertools
import sys
import os
from datetime import datetime

def run_experiment(model, opt, augmentation_type, bs, epochs, workers, dropout, weight_decay, lr):
    """Executes a single training run via main.py with augmentation configuration"""
    print(f"\n{'='*60}")
    print(f"RUNNING: Model={model.upper()} | Opt={opt.upper()}")
    print(f"Augmentation Type={augmentation_type.upper()}")
    print(f"{'='*60}")

    # Use virtual environment's python
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
        "--lr", str(lr),
        "--augmentation_type", augmentation_type,
    ]

    try:
        # Raise error if main.py crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Experiment Failed: {model} with {opt} and augmentation_type={augmentation_type}. Error: {e}")

def main():
    # 1. Define Grid
    models = ["vgg16", "resnet18", "efficientnet_b0"]    
    augmentation_types = ["none", "simple", "advanced", "both"]

    # Final winning parameters from Phase B
    best_params = {
        "vgg16": {"opt": "adamw", "bs": 64, "do": 0.2, "wd": 0.01},
        "resnet18": {"opt": "adamw", "bs": 32, "do": 0.1, "wd": 0.01},
        "efficientnet_b0": {"opt": "adamw", "bs": 32, "do": 0.2, "wd": 0.01}
    }

    # Static settings
    epochs = 10
    num_workers = 4
    learning_rate = 1e-4
    
    # 2. Generate combinations
    experiments = list(itertools.product(models, augmentation_types))
    
    total = len(experiments)
    print(f"Starting Final Evaluation (Baselines & Augmentations): {total} experiments queued.")
    start_time = datetime.now()

    # 3. Execute experiments
    for i, (model, aug_type) in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{total}")
        params = best_params[model]
        run_experiment(model, params["opt"], aug_type, params["bs"], epochs, num_workers, params["do"], params["wd"], learning_rate)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"DATA AUGMENTATION EXPERIMENTS COMPLETE in {duration}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
