import subprocess
import itertools
import sys
import os
from datetime import datetime

def run_experiment(model, opt, bs, epochs, workers, dropout, weight_decay, lr):
    """Executes a single training run via main.py"""
    print(f"\n{'='*60}")
    print(f"RUNNING: Model={model.upper()} | Opt={opt.upper()} | BS={bs} | LR={lr}")
    print(f"{'='*60}")

    # Use sys.executable to ensure we use the current virtual env's python
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--optimizer", opt,
        "--batch_size", str(bs),
        "--epochs", str(epochs),
        "--num_workers", str(workers),
        "--seed", "42", # Fixed seed for reproducibility
        "--dropout", str(dropout),
        "--weight_decay", str(weight_decay),
        "--skip_test", # Skip final evaluation during tuning
        "--lr", str(lr),
    ]

    try:
        # check=True will raise an error if main.py crashes
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Experiment Failed: {model} with {opt}. Error: {e}")

def main():
    # 1. Define your Grid Parameters
    models = ["vgg16", "resnet18", "efficientnet_b0"]
    optimizers = ["adamw", "sgd"]
    batch_sizes = [32, 64, 128]
    dropout_rates = [0.1, 0.2, 0.5]
    weight_decays = [1e-4, 1e-3, 1e-2]

    # Static settings
    epochs = 2  # Reduced to 2: enough to spot a trend while saving time
    num_workers = 4 # Set according to your device setup
    learning_rate = 1e-4 # Lowered from 0.001 to prevent shattering pre-trained weights
    
    # 2. Decoupled Grid Search (Prevents 162+ combinatorial explosion)
    
    # --- PHASE A: Find best training parameters (18 experiments) ---
    # Fixes regularization to baseline: Dropout=0.1, WD=1e-4
    experiments_phase_a = list(itertools.product(models, optimizers, batch_sizes, [0.1], [1e-4]))
    
    # --- PHASE B: Find best regularization parameters (27 experiments) ---
    # TODO: Update these two variables based on your Phase A results before running Phase B!
    best_opt = "adamw"   
    best_bs = 64         
    experiments_phase_b = list(itertools.product(models, [best_opt], [best_bs], dropout_rates, weight_decays))
    
    # CHOOSE WHICH PHASE TO RUN HERE:
    experiments = experiments_phase_a
    
    total = len(experiments)
    print(f"Starting Grid Search: {total} experiments queued.")
    start_time = datetime.now()

    # 3. Loop through and run
    for i, (model, opt, bs, dropout, weight_decay) in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{total}")
        run_experiment(model, opt, bs, epochs, num_workers, dropout, weight_decay, learning_rate)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE in {duration}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()