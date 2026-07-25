import os
import yaml

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from dataset import get_dataloaders
from model import BrainTumorModel


def evaluate():

    
    # CONFIGURACIÓ
    

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    classes = list(config["classes"].values())

    models_dir = config["paths"]["models"]

    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    
    # GPU
    

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDispositiu: {device}")

    
    # DATA
    

    _, _, test_loader = get_dataloaders()

    
    # MODEL
    

    model = BrainTumorModel(num_classes=4)

    model.load_state_dict(
        torch.load(
            os.path.join(models_dir, "best_model.pth"),
            map_location=device,
        )
    )

    model.to(device)

    model.eval()

    
    # PREDICCIONS
    

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())

            y_pred.extend(predictions.cpu().numpy())

    
    # ACCURACY
    

    accuracy = accuracy_score(y_true, y_pred)

    print("\n==============================")
    print("RESULTATS")
    print("==============================")

    print(f"\nAccuracy: {accuracy*100:.2f}%")

    
    # CLASSIFICATION REPORT
    

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
    )

    print("\nClassification Report\n")

    print(report)

    with open(
        os.path.join(results_dir, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    
    # CONFUSION MATRIX
    

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicció")

    plt.ylabel("Valor Real")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nConfusion Matrix guardada a:"
    )

    print(
        "results/confusion_matrix.png"
    )

    print(
        "\nClassification report guardat a:"
    )

    print(
        "results/classification_report.txt"
    )


if __name__ == "__main__":
    evaluate()