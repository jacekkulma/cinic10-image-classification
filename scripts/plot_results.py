import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_learning_curves(results_dir, plots_dir):
    """Generates Training vs Validation curves for each history JSON."""
    history_files = glob.glob(os.path.join(results_dir, "*_history.json"))
    
    for h_file in history_files:
        with open(h_file, 'r') as f:
            history = json.load(f)
            
        # Extract clean name for title
        base_name = os.path.basename(h_file).replace("_history.json", "")
        title_name = base_name.replace("_", " ").upper()
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Accuracy
        ax1.plot(epochs, history['train_acc'], 'b-', label='Training Acc', linewidth=2)
        ax1.plot(epochs, history['valid_acc'], 'r-', label='Validation Acc', linewidth=2)
        ax1.set_title(f'Accuracy: {title_name}')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Loss
        ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title(f'Loss: {title_name}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(plots_dir, f"{base_name}_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
def plot_model_comparison_curves(results_dir, plots_dir):
    """Generates comparison curves of the 3 models grouped by augmentation type."""
    history_files = glob.glob(os.path.join(results_dir, "*_history.json"))
    
    # Group files by augmentation type
    aug_groups = {"none": [], "simple": [], "advanced": [], "both": []}
    
    for h_file in history_files:
        base_name = os.path.basename(h_file).replace("_history.json", "")
        parts = base_name.split('_')
        aug = parts[-1].lower() # Augmentation is the last part of the name
        
        # Reconstruct model name
        model = "efficientnet_b0" if "efficientnet" in base_name else parts[0].lower()
            
        if aug in aug_groups:
            aug_groups[aug].append((model, h_file))
            
    colors = {"vgg16": "#1f77b4", "resnet18": "#ff7f0e", "efficientnet_b0": "#2ca02c"}
    
    for aug, files in aug_groups.items():
        if not files:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for model, h_file in files:
            with open(h_file, 'r') as f:
                history = json.load(f)
                
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors.get(model, "black")
            
            # Plot Accuracy (Solid for Val, Dashed for Train)
            ax1.plot(epochs, history['train_acc'], linestyle='--', color=color, alpha=0.4)
            ax1.plot(epochs, history['valid_acc'], linestyle='-', color=color, label=f'{model.upper()}', linewidth=2.5)
            
            # Plot Loss (Solid for Val, Dashed for Train)
            ax2.plot(epochs, history['train_loss'], linestyle='--', color=color, alpha=0.4)
            ax2.plot(epochs, history['valid_loss'], linestyle='-', color=color, label=f'{model.upper()}', linewidth=2.5)
            
        for ax, title, ylabel in zip([ax1, ax2], ['Accuracy', 'Loss'], ['Accuracy', 'Loss']):
            ax.set_title(f'{title} Comparison ({aug.upper()} Augmentation)')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(ylabel)
            ax.legend(title="Validation Metrics")
            ax.grid(True, linestyle='--', alpha=0.7)
            
        fig.text(0.5, 0.02, 'Solid lines = Validation | Dashed lines = Training', ha='center', fontsize=11, style='italic')
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        save_path = os.path.join(plots_dir, f"comparison_curves_{aug}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

def plot_augmentation_barchart(results_dir, plots_dir):
    """Generates a grouped bar chart comparing Final Test Accuracies."""
    txt_files = glob.glob(os.path.join(results_dir, "*_results.txt"))
    
    # Dictionary to hold our parsed data
    data = {"vgg16": {}, "resnet18": {}, "efficientnet_b0": {}}
    aug_types = ["none", "simple", "advanced", "both"]
    
    for t_file in txt_files:
        with open(t_file, 'r') as f:
            lines = f.readlines()
            
        parsed = {}
        for line in lines:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                parsed[k.strip()] = v.strip()
                
        # Plot final 10-epoch runs
        if parsed.get("Epochs") == "10" and parsed.get("Test Accuracy") != "Skipped":
            model = parsed["Model"].lower()
            aug = parsed["Augmentation"].lower()
            acc = float(parsed["Test Accuracy"].replace("%", ""))
            data[model][aug] = acc
            
    # Setup Bar Chart
    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bars for each augmentation type
    for i, aug in enumerate(aug_types):
        values = [data[m].get(aug, 0.0) for m in models]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=aug.capitalize())
        # Add labels on top of bars
        ax.bar_label(rects, padding=3, fmt='%.1f%%', fontsize=9)

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Impact of Data Augmentation Across Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(title="Augmentation Type")
    ax.set_ylim(40, 85) # Adjust limits for better scaling
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "augmentation_comparison_barchart.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_few_shot_barchart(results_dir, plots_dir):
    """Generates a stacked bar chart comparing Few-Shot and Zero-Shot evaluations."""
    fs_files = glob.glob(os.path.join(results_dir, "few_shot_*.json"))
    zs_files = glob.glob(os.path.join(results_dir, "zero_shot_*.json"))
        
    if not fs_files and not zs_files:
        return  # Skip if evaluation not done yet
        
    # Pre-define our 3 models to ensure they align perfectly
    data = {"vgg16": {}, "resnet18": {}, "efficientnet_b0": {}}
    
    # Helper function to match the file to the correct model bucket
    def get_model_key(filename):
        fname = os.path.basename(filename).lower()
        if "efficientnet" in fname: return "efficientnet_b0"
        if "resnet18" in fname or "resnet-18" in fname: return "resnet18"
        if "vgg16" in fname or "vgg-16" in fname: return "vgg16"
        return None

    # Parse Few-Shot (Fine-Tuned) Data
    for f in fs_files:
        key = get_model_key(f)
        if key:
            with open(f, 'r') as file:
                d = json.load(file)
                # Handle standard float (0.722) to percentage (72.2)
                acc = d.get('mean_accuracy', 0.0)
                data[key]['few_shot'] = acc * 100 if acc <= 1.0 else acc
                
    # Parse Zero-Shot (Pre-trained Baseline) Data
    for f in zs_files:
        key = get_model_key(f)
        if key:
            with open(f, 'r') as file:
                d = json.load(file)
                acc = d.get('mean_accuracy', 0.0)
                data[key]['zero_shot'] = acc * 100 if acc <= 1.0 else acc
                
    models = [m for m in data.keys() if data[m]]
    if not models:
        return
        
    x = np.arange(len(models))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract values safely
    zs_values = [data[m].get('zero_shot', 0.0) for m in models]
    fs_values = [data[m].get('few_shot', 0.0) for m in models]
    
    # Calculate the "Gain" (the blue top part of the stacked bar)
    fs_gains = [max(0, fs - zs) for fs, zs in zip(fs_values, zs_values)]
    
    # Plot Bottom Bar (Zero-Shot Baseline - Orange)
    rects1 = ax.bar(x, zs_values, width, label='Zero-Shot Baseline', color='#ff7f0e', edgecolor='white')
    
    # Plot Top Bar (Gain from Fine-Tuning - Blue) stacked on top of zs_values
    rects2 = ax.bar(x, fs_gains, width, bottom=zs_values, label='Gain from Fine-Tuning', color='#1f77b4', edgecolor='white')
    
    # Add labels inside the bottom bar (Zero-Shot accuracy)
    ax.bar_label(rects1, label_type='center', fmt='%.1f%%', color='white', weight='bold')
    
    # Add labels at the very top of the stacked bar (Total Fine-Tuned accuracy)
    ax.bar_label(rects2, labels=[f"{fs:.1f}%" for fs in fs_values], padding=3, color='black', weight='bold', fontsize=11)
    
    # Formatting
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_title('Zero-Shot vs Fine-Tuned 5-Shot SimpleShot Performance')
    ax.set_xticks(x)
    
    # Clean up model names for the x-axis
    formatted_models = [m.replace('_', '-').upper() for m in models]
    ax.set_xticklabels(formatted_models)
    
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "few_shot_barchart.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints')
    base_results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plots_dir = os.path.join(base_results_dir, 'plots')
    ensure_dir(plots_dir)
    
    print("Generating Learning Curves...")
    plot_learning_curves(results_dir, plots_dir)
    print("Generating Model Comparison Curves...")
    plot_model_comparison_curves(results_dir, plots_dir)
    print("Generating Augmentation Comparison Bar Chart...")
    plot_augmentation_barchart(results_dir, plots_dir)
    print("Generating Few-Shot Bar Chart...")
    plot_few_shot_barchart(base_results_dir, plots_dir)
    print(f"Done! All plots saved to: {plots_dir}")
