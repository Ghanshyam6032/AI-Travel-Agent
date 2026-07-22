# Python ka lightweight base image use karein
FROM python:3.10-slim

# Container ke andar working directory set karein
WORKDIR /app

# Requirements file copy karein aur libraries install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aapka code (main.py, index.html) container mein copy karein
COPY . .

# Port 8000 expose karein (FastAPI default port)
EXPOSE 8000

# Server ko start karne ka command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]