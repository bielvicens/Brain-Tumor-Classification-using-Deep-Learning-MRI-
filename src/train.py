import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataset import get_dataloaders
from model import BrainTumorModel


def evaluate(model, dataloader, criterion, device):
    """
    Avalua el model sobre qualsevol DataLoader.
    Retorna:
        loss mitjana
        accuracy
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train():

    
    # CONFIGURACIÓ
    

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    models_dir = config["paths"]["models"]

    os.makedirs(models_dir, exist_ok=True)

    
    # DISPOSITIU
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Entrenant amb: {device}")

    
    # DADES
    

    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"\nTrain:      {len(train_loader.dataset)} imatges")
    print(f"Validation: {len(val_loader.dataset)} imatges")
    print(f"Test:       {len(test_loader.dataset)} imatges")

    
    # MODEL
    

    model = BrainTumorModel(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    
    # ENTRENAMENT
    

    # Inicialització per a l'Early Stopping
    best_val_acc = 0.0
    # Historial d'entrenament
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []
    patience = 5
    counter = 0

    for epoch in range(epochs):

        print(f"\n==============================")
        print(f"ÈPOCA {epoch+1}/{epochs}")
        print(f"==============================")

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        
        # VALIDACIÓ
        

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # Actualitzem el learning rate scheduler
        scheduler.step(val_acc)

        
        # RESULTATS
        

        print(f"\nTrain Loss      : {train_loss:.4f}")
        print(f"Train Accuracy  : {train_acc:.2f}%")

        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Validation Acc. : {val_acc:.2f}%")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        
        # GUARDAR MILLOR MODEL & EARLY STOPPING
        

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            counter = 0 # Resetejem el comptador perquè hem millorat!

            torch.save(
                model.state_dict(),
                os.path.join(models_dir, "best_model.pth")
            )

            print("\n Nou millor model guardat!")
            
        else:
            counter += 1
            print(f"\n Sense millora. Comptador d'Early Stopping: {counter}/{patience}")
            
            if counter >= patience:
                print("\n Early stopping!")
                break # Tallem el bucle de les èpoques

    
    # TEST FINAL
    

    print("\n===================================")
    print("Carregant el millor model...")
    print("===================================\n")

    model.load_state_dict(
        torch.load(
            os.path.join(models_dir, "best_model.pth"),
            map_location=device
        )
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n===================================")
    print("RESULTAT FINAL")
    print("===================================")

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.2f}%")

    print("\nEntrenament finalitzat correctament.")
    ############################################################
    # GUARDAR GRÀFIQUES
    ############################################################

    os.makedirs("results", exist_ok=True)

    epochs_range = range(1, len(train_losses)+1)

    # LOSS

    plt.figure(figsize=(8,5))

    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/loss.png")

    plt.close()

    # ACCURACY

    plt.figure(figsize=(8,5))

    plt.plot(epochs_range, train_accuracies, label="Train")
    plt.plot(epochs_range, val_accuracies, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/accuracy.png")

    plt.close()


if __name__ == "__main__":
    train()