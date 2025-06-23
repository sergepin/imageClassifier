# 🐱🐶 Image Classifier - Cat vs Dog

Un clasificador de imágenes que distingue entre gatos y perros utilizando deep learning con PyTorch y una API web con FastAPI.

## 📋 Descripción

Este proyecto implementa un clasificador de imágenes que utiliza un modelo ResNet18 pre-entrenado para clasificar imágenes entre dos categorías: gatos y perros. Incluye:

- **Entrenamiento del modelo**: Script para entrenar el clasificador con tu propio dataset
- **API REST**: Servidor FastAPI para hacer predicciones en tiempo real
- **Interfaz web**: Endpoint para subir imágenes y obtener predicciones

## 🏗️ Estructura del Proyecto

```
imageClassifier/
├── api/
│   └── app.py              # Servidor FastAPI
├── model/
│   └── train.py            # Script de entrenamiento
├── dataset/
│   ├── Cat/                # Imágenes de gatos
│   └── Dog/                # Imágenes de perros
├── venv/                   # Entorno virtual
├── requirements.txt        # Dependencias del proyecto
└── readme.md               # Este archivo
```

## 🚀 Instalación

### 1. Crear entorno virtual

```bash
python -m venv venv
```

### 2. Activar entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

**Opción A: Usar requirements.txt (Recomendado)**
```bash
pip install -r requirements.txt
```

**Opción B: Instalación manual**
```bash
# PyTorch con soporte CUDA (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencias adicionales
pip install fastapi uvicorn python-multipart pillow matplotlib numpy
```

### 4. Verificar instalación CUDA

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

## 📦 Dependencias Explicadas

### Core Dependencies
- **`torch`**: Framework principal de deep learning (PyTorch) con soporte CUDA
- **`torchvision`**: Utilidades para visión por computadora con PyTorch
  - Incluye modelos pre-entrenados (ResNet18)
  - Transformaciones de imágenes
  - Datasets y DataLoaders
- **`torchaudio`**: Utilidades para procesamiento de audio (incluido para compatibilidad)

### Web Framework
- **`fastapi`**: Framework web moderno y rápido para crear APIs
- **`uvicorn`**: Servidor ASGI para ejecutar aplicaciones FastAPI
- **`python-multipart`**: Manejo de archivos multipart (necesario para subir imágenes)

### Image Processing
- **`pillow`**: Biblioteca para procesamiento de imágenes (PIL)
- **`matplotlib`**: Visualización de datos y gráficos
- **`numpy`**: Computación numérica (dependencia de PyTorch)

## 🎯 Uso

### 1. Preparar el Dataset

Organiza tus imágenes en la siguiente estructura:
```
dataset/
├── Cat/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── Dog/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

### 2. Entrenar el Modelo

```bash
cd model
python train.py
```

El script:
- Carga el dataset desde `../dataset/`
- Entrena un modelo ResNet18 por 5 épocas
- Utiliza GPU si está disponible (CUDA)
- Guarda el modelo entrenado como `model.pth`

### 3. Ejecutar la API

```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Hacer Predicciones

La API estará disponible en `http://localhost:8000`

**Endpoint:** `POST /predict`

**Uso con curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@tu_imagen.jpg"
```

**Respuesta:**
```json
{
  "prediction": "cat"
}
```

## 🔧 Configuración Avanzada

### Modificar Hiperparámetros

En `model/train.py` puedes ajustar:
- `EPOCHS`: Número de épocas de entrenamiento
- `batch_size`: Tamaño del batch (actualmente 32)
- `lr`: Learning rate (actualmente 0.001)

### Cambiar el Modelo

Para usar un modelo diferente, modifica en `model/train.py`:
```python
# Cambiar ResNet18 por otro modelo
model = models.resnet50(weights=ResNet18_Weights.DEFAULT)
```

## 📊 Características del Modelo

- **Arquitectura**: ResNet18 pre-entrenado
- **Input**: Imágenes RGB de 224x224 píxeles
- **Output**: Clasificación binaria (gato/perro)
- **Optimizador**: Adam
- **Loss Function**: CrossEntropyLoss
- **GPU Support**: Compatible con CUDA para entrenamiento acelerado

## 🛠️ Comandos Útiles

### Verificar instalación de PyTorch
```bash
python -c "import torch; print(torch.__version__)"
```

### Verificar CUDA (GPU)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Verificar versión de CUDA
```bash
python -c "import torch; print(torch.version.cuda)"
```

### Documentación automática de la API
Una vez ejecutada la API, visita:
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### Error: "CUDA out of memory"
Reduce el `batch_size` en `train.py`

### Error: "Model file not found"
Asegúrate de haber entrenado el modelo antes de ejecutar la API

### Error: "CUDA not available"
Si tienes GPU NVIDIA pero CUDA no está disponible:
1. Instala los drivers NVIDIA más recientes
2. Instala CUDA Toolkit 11.8
3. Reinstala PyTorch con soporte CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Versiones de CUDA compatibles
- CUDA 11.8 (recomendado)
- CUDA 12.1
- CPU-only (sin GPU)

---

**¡Disfruta clasificando imágenes de gatos y perros! 🐾**
