import argparse
import torch
import os
import time
import json

# Import our custom modules
from src.utils import set_seed
from src.dataset import get_dataloaders
from src.models import get_model
from src.train import train_model, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="CINIC-10 Image Classification Grid Search")
    
    # Environment arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the CINIC-10 dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, choices=["vgg16", "resnet18", "efficientnet_b0"], help="Architecture to train")
    
    # Training process hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, choices=[32, 64, 128], help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--skip_test", action="store_true", help="Skip final evaluation on test set")
    
    # Regularization hyperparameters
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in classifier head")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 regularization coefficient")

    # Optional argument for data augmentation
    parser.add_argument("--augmentation_type", type=str, default="none", choices=["none", "simple", "advanced", "both"], help="Type of data augmentation to apply: 'none', 'simple', 'advanced' (Mixup), or 'both'")

    return parser.parse_args()

def main():
    # 0. Start timer
    start_time = time.time()

    # 1. Parse CLI arguments
    args = parse_args()
    
    # 2. Lock the random seed for reproducibility
    set_seed(args.seed)
    
    # 3. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Experiment ---")
    print(f"Device: {device}")
    print(f"Model: {args.model.upper()}")
    print(f"Batch Size: {args.batch_size} | Optimizer: {args.optimizer.upper()}")
    print(f"Dropout: {args.dropout} | Weight Decay: {args.weight_decay}")    
    print(f"Data Augmentation: {args.augmentation_type.capitalize()}")

    # 4. Load Data
    print("\nLoading data...")
    train_loader, valid_loader, test_loader = get_dataloaders(
        args.data_dir, 
        args.batch_size, 
        num_workers=args.num_workers,
        augmentation_type=args.augmentation_type,
        seed=args.seed
    )
    
    # 5. Initialize Model
    print("Initializing model...")
    model = get_model(args.model, num_classes=10, dropout_rate=args.dropout)
    
    # 6. Train Model
    print("Beginning training loop...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # 7. Save Checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    save_path = f"results/checkpoints/{args.model}_{args.optimizer}_bs{args.batch_size}_do{args.dropout}_{args.augmentation_type}.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"\nTraining complete. Model weights saved to: {save_path}")

    # 8. Final Evaluation on Test Set
    test_acc_str = "Skipped"
    if not args.skip_test:
        print("\n--- Final Evaluation ---")
        test_acc = evaluate_model(trained_model, test_loader, device)
        test_acc_str = f"{test_acc:.2f}%"
        print(f"Final Test Accuracy: {test_acc_str}")
        
    end_time = time.time()
    duration_mins = (end_time - start_time) / 60.0
    
    results_file = save_path.replace(".pth", "_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Augmentation: {args.augmentation_type}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"--------------------\n")
        f.write(f"Final Train Acc: {history['train_acc'][-1]*100:.2f}%\n")
        f.write(f"Final Valid Acc: {history['valid_acc'][-1]*100:.2f}%\n")
        f.write(f"--------------------\n")
        f.write(f"Test Accuracy: {test_acc_str}\n")
        f.write(f"Total Time: {duration_mins:.2f} minutes\n")

    # Save history for plotting
    history_file = save_path.replace(".pth", "_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Total Time: {duration_mins:.2f} minutes")
    print(f"Results logged to: {results_file}")
    print(f"Training history saved to: {history_file}")

if __name__ == "__main__":
    main()