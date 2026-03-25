import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(model, optimizer_name: str, lr: float, weight_decay: float):
    """
    Returns the specified optimizer initialized with the model's parameters.
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "sgd":
        # Using SGD with a standard momentum of 0.9 as per project plan
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported. Choose sgd or adamw.")

def train_model(model, train_loader, valid_loader, epochs: int, optimizer_name: str, lr: float, weight_decay: float, device: str):
    """
    The main training loop that updates model weights and evaluates on the validation set.
    """
    model = model.to(device)
    
    # CrossEntropyLoss is the standard loss function for multi-class classification (10 classes)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the optimizer you want to test
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)

    # We will track these to see how the hyper-parameters affect convergence
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }

    print(f"Starting training on {device} using {optimizer_name.upper()} optimizer...")

    for epoch in range(epochs):
        # -------------------
        # 1. Training Phase
        # -------------------
        model.train() # Set model to training mode (enables Dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_data in train_loader:
            # Check if this is a Mixup batch (4 elements) or regular batch (2 elements)
            if len(batch_data) == 4:
                # Mixup case: (images, labels_a, labels_b, alpha)
                images, labels_a, labels_b, alpha = batch_data
                images = images.to(device)
                labels_a = labels_a.to(device)
                labels_b = labels_b.to(device)
                alpha = alpha.to(device)
                
                # Zero the gradients from the previous step
                optimizer.zero_grad()

                # Forward pass: predict the classes
                outputs = model(images)
                
                # Mixup loss: weighted combination of two label losses
                loss = alpha * criterion(outputs, labels_a) + (1 - alpha) * criterion(outputs, labels_b)

                # Backward pass: calculate gradients
                loss.backward()

                # Optimizer step: update weights
                optimizer.step()

                # Track metrics (use labels_a for accuracy with Mixup)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels_a.size(0)
                correct_train += (predicted == labels_a).sum().item()
            else:
                # Regular case: (images, labels)
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)

                # Zero the gradients from the previous step
                optimizer.zero_grad()

                # Forward pass: predict the classes
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass: calculate gradients
                loss.backward()

                # Optimizer step: update weights
                optimizer.step()

                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # -------------------
        # 2. Validation Phase
        # -------------------
        model.eval() # Set model to evaluation mode (disables Dropout)
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        # Disable gradient calculation for validation to save memory and compute
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        epoch_valid_loss = running_valid_loss / total_valid
        epoch_valid_acc = correct_valid / total_valid

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['valid_loss'].append(epoch_valid_loss)
        history['valid_acc'].append(epoch_valid_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}")

    return model, history

def evaluate_model(model, test_loader, device):
    """
    Runs evaluation on test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): # Critical: disables gradient tracking to save memory
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy