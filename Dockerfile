FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/siva

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENCV_AVOIDSTRFTIME=1

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 --preload --worker-class gthread --worker-connections 1000 main:app