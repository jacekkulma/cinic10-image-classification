import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int = 10, dropout_rate: float = None):
    """
    Loads pre-trained models from torchvision and modifies their 
    classifier heads to output the specified number of classes.
    """
    model_name = model_name.lower()

    if model_name == "vgg16":
        # Deep CNN baseline
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Indexes 2 and 5 are the default Dropout layers.
        if dropout_rate is not None:
            model.classifier[2].p = dropout_rate
            model.classifier[5].p = dropout_rate
            
        # Index 6 is the final Linear layer
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        # Residual network
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ResNet-18's final layer is simply called 'fc' (fully connected)
        in_features = model.fc.in_features
        
        # Inject dropout if provided
        if dropout_rate is not None:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Compound scaling model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Index 0 is the Dropout layer, Index 1 is the Linear layer.
        if dropout_rate is not None:
            model.classifier[0].p = dropout_rate
            
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from vgg16, resnet18, efficientnet_b0.")

    return model