FROM python:3.11-slim

# Evita preguntas interactivas al instalar paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias necesarias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crea el directorio de trabajo
WORKDIR /app

# Copia primero solo los requirements (para aprovechar el cache de docker)
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

# Expone el puerto por donde se levantará la app
EXPOSE 8080

# Comando para ejecutar la app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
