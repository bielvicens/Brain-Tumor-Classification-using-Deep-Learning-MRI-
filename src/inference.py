import sys
import yaml

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import BrainTumorModel



# CONFIG


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

img_size = config["training"]["img_size"]

classes = config["classes"]


# DEVICE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositiu: {device}")


# MODEL


model = BrainTumorModel(num_classes=4)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()


# TRANSFORM


transform = transforms.Compose([

    transforms.Resize((img_size, img_size)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


# IMAGE


if len(sys.argv) != 2:

    print("Ús:")
    print("python src/inference.py imatge.jpg")

    sys.exit()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# PREDICCIÓ


with torch.no_grad():

    outputs = model(image)

    probabilities = F.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities,1)

predicted_class = classes[int(prediction)]

print("\n==============================")
print("PREDICCIÓ")
print("==============================\n")

print(f"Classe     : {predicted_class}")
print(f"Confiança  : {confidence.item()*100:.2f}%")