import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

MODEL_PATH = "models/resnet50_preprocessed_best.pth"
DEVICE = torch.device("cpu")

# Mismas transformaciones usadas para validación en entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict(image_path):
    print(f"\nAnalizando imagen: {image_path}\n")

    model = load_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = "Melanoma (1)" if pred_class == 1 else "Benigna (0)"

    print("Resultado de predicción:")
    print(f" - Clase: {label}")
    print(f" - Confianza: {confidence*100:.2f}%\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_image_preprocessed.py ruta_imagen.jpg")
        sys.exit(1)

    predict(sys.argv[1])
