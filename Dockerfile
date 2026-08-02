# Python ka lightweight base image use karein
FROM python:3.10-slim

# Container ke andar working directory set karein
WORKDIR /app

# Requirements file copy karein aur libraries install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sirf backend code copy karein (Kyunki frontend ab GitHub Pages par hai)
COPY main.py .

# Railway dynamic port ($PORT) assign karta hai, isliye environment variable use hoga
EXPOSE 8080

# Server ko start karne ka command ($PORT ko support karte hue)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}