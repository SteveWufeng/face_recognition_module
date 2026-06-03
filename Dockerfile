FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY pyproject.toml .
COPY face_recognition/ face_recognition/
RUN pip install --no-cache-dir .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# RealSense SDK — provides the UVC backend so OpenCV can read RGB from RealSense cameras
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://librealsense.intel.com/Debian/librealsense.pgp | gpg --dearmor -o /usr/share/keyrings/librealsense-archive.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/librealsense-archive.gpg] https://librealsense.intel.com/Debian/apt-repo bookworm main" \
        > /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        librealsense2-udev-rules \
        librealsense2-gl \
        librealsense2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/face-recognition /usr/local/bin/face-recognition
COPY --from=builder /app/face_recognition /app/face_recognition

WORKDIR /workspace

ENTRYPOINT ["face-recognition"]
CMD ["--help"]
