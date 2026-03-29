# CINIC-10 Image Classification with CNNs

This repository contains the source code for evaluating Convolutional Neural Network architectures (VGG-16, ResNet-18, and EfficientNet-B0) on the CINIC-10 dataset. The pipeline includes hyperparameter tuning, data augmentation analysis (including Mixup), and Few-Shot learning evaluation.

---

## ⚙️ Step 1: Environment Setup

All experiments require **Python 3.13** or newer. It is highly recommended to use a virtual environment to prevent dependency conflicts.

### 1. Create and Activate a Virtual Environment

**For Linux/macOS users:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows users:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies (CPU vs. GPU Training)
The project provides two separate dependency files depending on your hardware capabilities:

* **GPU (Recommended):** If you have an NVIDIA GPU, you must install the CUDA-enabled versions of PyTorch to ensure training finishes in a reasonable time.
  ```bash
  pip install -r requirements-cuda.txt
  ```
* **CPU (Fallback):** If you are running on a standard laptop or a machine without a dedicated NVIDIA GPU, use the standard requirements file.
  ```bash
  pip install -r requirements.txt
  ```

---

## 📂 Step 2: Data Preparation

The CINIC-10 dataset is not included in the source code due to its size.

1. Download the CINIC-10 dataset from the official Kaggle or academic repository.
2. Extract the dataset into a folder named `data/` at the root of the project directory.
3. Ensure the directory structure exactly matches the following format so the PyTorch `ImageFolder` loaders can parse it:
   * `data/train/`
   * `data/valid/`
   * `data/test/`

---

## 🚀 Step 3: Executing the Pipeline

The project relies on a modular script architecture. Commands should be run from the root directory of the project.

### 1. Hyperparameter Grid Search
To execute the decoupled grid search (Phase A and Phase B), run the automated script. You must manually toggle the `experiments` variable inside the script to switch between phases.
```bash
python scripts/run_grid_search.py
```

### 2. Full Model Training & Data Augmentation
To automatically train all three architectures across all four data augmentation strategies (yielding 12 final models), execute the augmentation orchestrator script. This script utilizes the optimal hyperparameters discovered during the grid search phase:
```bash
python scripts/run_data_augmentation.py
```
*Note: If replicating this pipeline on a new dataset, ensure the `best_params` dictionary inside this script is updated with your new grid search results before executing.*

### 3. Few-Shot Evaluation
To evaluate a fine-tuned model checkpoint on the 10-way 5-shot task using the SimpleShot algorithm, execute the few-shot script, passing the path to the trained `.pth` weights:
```bash
python scripts/run_few_shot.py \
    --model vgg16 \
    --checkpoint results/checkpoints/vgg16_adamw_bs64_do0.2_advanced.pth \
    --dropout 0.2
```

### 4. Generating Visualizations
Once all training logs (`_history.json`) and evaluation files (`_results.txt`) have been generated in the `results/` directory, the analytical plots used in the report can be automatically generated via:
```bash
python scripts/plot_results.py
```
All plots will be saved to the `results/plots/` directory.
