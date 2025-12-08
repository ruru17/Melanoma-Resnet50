import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# ------- Configuración -------
MODEL_PATH = "models/resnet50_melanoma_best.pth"
DEVICE = torch.device("cpu")

# ------- Transformaciones ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------- Cargar modelo -------
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 clases: benigno / melanoma

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ------- Predecir -------
def predict(image_path):
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    label = pred.item()
    conf = confidence.item() * 100

    print("\n===================================")
    print(f"Imagen: {image_path}")
    print("-----------------------------------")

    if label == 1:
        print("🔴 PREDICCIÓN: MELANOMA (1)")
    else:
        print("🟢 PREDICCIÓN: BENIGNO (0)")

    print(f"Confianza del modelo: {conf:.2f}%")
    print("===================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_image.py ruta_imagen.jpg")
        sys.exit(1)

    predict(sys.argv[1])
