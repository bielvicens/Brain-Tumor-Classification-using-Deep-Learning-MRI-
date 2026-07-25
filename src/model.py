import torch
import torch.nn as nn
from torchvision import models


class BrainTumorModel(nn.Module):
    """
    ResNet50 preentrenada amb ImageNet adaptada
    per a la classificació de tumors cerebrals.

    Estratègia:
        - Layers 1,2,3 congelades
        - Layer4 entrenable
        - FC entrenable
    """

    def __init__(self, num_classes=4):

        super().__init__()

        
        # Carregar ResNet50 preentrenada
        

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        
        # Congelar totes les capes
        

        for param in self.model.parameters():
            param.requires_grad = False

        
        # Descongelar només layer4
        

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        
        # Substituir la capa final
        

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Dropout(p=0.5),

            nn.Linear(num_features, num_classes)

        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = BrainTumorModel()

    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n===============================")
    print(f"Paràmetres totals      : {total_params:,}")
    print(f"Paràmetres entrenables : {trainable_params:,}")
    print("===============================\n")

    dummy = torch.randn(8, 3, 224, 224)

    output = model(dummy)

    print("Entrada :", dummy.shape)
    print("Sortida :", output.shape)