import argparse
import torch
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.few_shot import evaluate_few_shot
from src.utils import set_seed
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# CINIC-10 specific statistics
CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


def parse_args():
    parser = argparse.ArgumentParser(description="SimpleShot Few-Shot Learning Evaluation")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, 
                       choices=["vgg16", "resnet18", "efficientnet_b0"],
                       help="Model architecture to evaluate")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Path to dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"],
                       help="Which split to evaluate on")
    
    # Few-shot task parameters
    parser.add_argument("--n_way", type=int, default=10,
                       help="Number of classes in few-shot task")
    parser.add_argument("--k_shot", type=int, default=5,
                       help="Number of support samples per class")
    parser.add_argument("--q_shot", type=int, default=1,
                       help="Number of query samples per class")
    parser.add_argument("--n_episodes", type=int, default=100,
                       help="Number of few-shot episodes to evaluate")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="./results/few_shot_results.json",
                       help="Path to save results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading pre-trained model: {args.model}")
    model = get_model(args.model, num_classes=10)
    model.to(device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)
    ])
    
    dataset_path = os.path.join(args.data_dir, args.split)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Run few-shot evaluation
    print(f"\n--- Few-Shot Evaluation ---")
    print(f"Task: {args.n_way}-way {args.k_shot}-shot {args.q_shot}-query")
    print(f"Episodes: {args.n_episodes}")
    print(f"Running evaluation...\n")
    
    results = evaluate_few_shot(
        model=model,
        model_name=args.model,
        dataset=dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_shot=args.q_shot,
        n_episodes=args.n_episodes,
        device=device
    )
    
    # Display results
    print(f"--- Results ---")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std Accuracy:  {results['std_accuracy']:.4f}")
    print(f"Successful Episodes: {results['n_episodes']}/{args.n_episodes}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()