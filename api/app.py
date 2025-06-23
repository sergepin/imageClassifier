from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

app = FastAPI()

# Clases
class_names = ['cat', 'dog']

# Modelo
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("../model/model.pth", map_location=torch.device('cpu')))
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # batch size 1

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            prediction = class_names[pred.item()]

        return {"prediction": prediction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
