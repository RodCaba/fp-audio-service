#!/bin/bash

# Generate Python gRPC code from proto files
# Audio service only needs audio_service.proto since it serves the gRPC API

python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./src/grpc/grpc_generated \
    --grpc_python_out=./src/grpc/grpc_generated \
    ./proto/audio_service.proto

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "Error generating gRPC code. Please check your proto files and try again."
    exit 1
fi

# Fix import paths in generated files
sed -i 's/import audio_service_pb2/from . import audio_service_pb2/g' ./src/grpc/grpc_generated/audio_service_pb2_grpc.py

echo "gRPC code generated successfully for Audio service (gRPC server)!"
