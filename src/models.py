import torch.nn as nn
from torchvision import models

def get_model(model_name: str, num_classes: int = 10, dropout_rate: float = None):
    """
    Loads pre-trained models from torchvision and modifies their 
    classifier heads to output the specified number of classes.
    """
    model_name = model_name.lower()

    if model_name == "vgg16":
        # Load the standard deep CNN baseline
        # We use the modern 'weights' parameter instead of the deprecated 'pretrained=True'
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # VGG-16's classifier is a Sequential block. 
        # Indexes 2 and 5 are the default Dropout layers.
        if dropout_rate is not None:
            model.classifier[2].p = dropout_rate
            model.classifier[5].p = dropout_rate
            
        # Index 6 is the final Linear layer (originally out_features=1000 for ImageNet)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        # Load the residual network [cite: 11]
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ResNet-18's final layer is simply called 'fc' (fully connected)
        in_features = model.fc.in_features
        
        # ResNets don't use Dropout by default. If we want to test it, we must inject it.
        if dropout_rate is not None:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        # Load the compound scaling model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # EfficientNet's classifier is a Sequential block.
        # Index 0 is the Dropout layer, Index 1 is the Linear layer.
        if dropout_rate is not None:
            model.classifier[0].p = dropout_rate
            
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from vgg16, resnet18, efficientnet_b0.")

    return model