# Multi-stage build for ARM64/AMD64 compatibility
FROM python:3.11-slim AS builder

# Install build dependencies, add audio libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libjack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM builder

# Install runtime dependencies and audio libraries
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libasound2 \
    libasound2-plugins \
    libpulse0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create audio group and non-root user for security
RUN groupadd -r audio || true && \
    groupadd -r app || true && \
    useradd --create-home --shell /bin/bash -g audio -G app app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY . .

# Copy only the required proto file
COPY ./proto/audio_service.proto /tmp/audio_service.proto

# Create necessary directories
RUN mkdir -p data/recorded_audio data/audio exported_models src/grpc_generated && \
    chown -R app:app /app

# Install kitchen20 and common libraries
RUN pip install -e ./lib/kitchen20-pytorch
RUN pip install -e ./lib/common

# Generate gRPC code at build time
RUN python -m grpc_tools.protoc \
    --proto_path=/tmp \
    --python_out=./src/grpc_generated \
    --grpc_python_out=./src/grpc_generated \
    /tmp/audio_service.proto && \
    sed -i 's/import audio_service_pb2/from . import audio_service_pb2/g' ./src/grpc_generated/audio_service_pb2_grpc.py

# Create audio configuration
RUN mkdir -p /home/app/.config/pulse && \
    echo "default-server = unix:/run/user/1000/pulse/native" > /home/app/.config/pulse/client.conf && \
    chown -R app:audio /home/app/.config

# Switch to non-root user
USER app

# Configure audio environment
ENV PULSE_RUNTIME_PATH=/run/user/1000/pulse
ENV PULSE_COOKIE=/home/app/.config/pulse/cookie

# Add local packages to PATH
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app/src

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; from src.grpc_generated import audio_service_pb2_grpc, audio_service_pb2; \
    channel = grpc.insecure_channel('localhost:50051'); \
    stub = audio_service_pb2_grpc.AudioServiceStub(channel); \
    stub.HealthCheck(audio_service_pb2.HealthCheckRequest())" || exit 1

# Run the gRPC server
CMD ["python", "app.py"]