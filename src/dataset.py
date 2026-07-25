import os
import yaml
from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """
    Dataset personalitzat per carregar les imatges MRI.
    """

    def __init__(self, data_dir, config_classes, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # Convertim {0:"glioma"} -> {"glioma":0}
        self.class_to_idx = {
            name: idx for idx, name in config_classes.items()
        }

        for class_name, idx in self.class_to_idx.items():

            class_dir = os.path.join(data_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):

                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):

                    self.image_paths.append(
                        os.path.join(class_dir, img_name)
                    )

                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(config_path="configs/config.yaml"):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = config["training"]["img_size"]
    batch_size = config["training"]["batch_size"]

    raw_data_path = config["paths"]["raw_data"]

    classes = config["classes"]

    
    # TRANSFORMACIONS
    

    train_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.RandomRotation(5),

        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_test_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    
    # DATASET TRAIN (SENSE AUGMENTATION)
    

    base_dataset = BrainTumorDataset(
        data_dir=os.path.join(raw_data_path, "Training"),
        config_classes=classes,
        transform=None
    )

    
    # SPLIT ESTRATIFICAT
    

    indices = list(range(len(base_dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=base_dataset.labels
    )

    
    # DATASET TRAIN
    

    train_dataset = BrainTumorDataset(
        data_dir=os.path.join(raw_data_path, "Training"),
        config_classes=classes,
        transform=train_transform
    )

    
    # DATASET VALIDATION
    

    val_dataset = BrainTumorDataset(
        data_dir=os.path.join(raw_data_path, "Training"),
        config_classes=classes,
        transform=val_test_transform
    )

    
    # APLIQUEM ELS ÍNDEXS
    

    train_dataset = Subset(train_dataset, train_idx)

    val_dataset = Subset(val_dataset, val_idx)

    
    # DATASET TEST
    

    test_dataset = BrainTumorDataset(
        data_dir=os.path.join(raw_data_path, "Testing"),
        config_classes=classes,
        transform=val_test_transform
    )

    
    # DATALOADERS
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print("\n========== DATASET ==========\n")

    print(f"Train:      {len(train_loader.dataset)} imatges")

    print(f"Validation: {len(val_loader.dataset)} imatges")

    print(f"Test:       {len(test_loader.dataset)} imatges")

    images, labels = next(iter(train_loader))

    print(f"\nBatch imatges: {images.shape}")

    print(f"Batch labels: {labels.shape}")