FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ros-jazzy-rmw-zenoh-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir \
    insightface>=1.0.1 \
    opencv-python-headless>=4.8 \
    numpy>=1.24

WORKDIR /app
COPY face_recognition/ face_recognition/
COPY pyproject.toml .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
