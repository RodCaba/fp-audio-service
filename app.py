import grpc
from concurrent import futures
import time
import os

from grpc_reflection.v1alpha import reflection

from src.audio_service import AudioService
from src.grpc_generated import audio_service_pb2_grpc


def serve():
    """Start the gRPC server"""
    port = os.environ.get('GRPC_PORT', '50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    audio_service_pb2_grpc.add_AudioServiceServicer_to_server(
        AudioService(), server
    )

    # Enable gRPC reflection
    SERVICE_NAMES = (
        'audio_service.AudioService',  # Use the full service name from proto
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    print(f"Starting Audio Service gRPC server on {listen_addr}")
    server.start()
    
    try:
        while True:
            time.sleep(86400)  # Keep server running
    except KeyboardInterrupt:
        print("Shutting down Audio Service...")
        server.stop(0)


if __name__ == '__main__':
    serve()
