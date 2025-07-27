from pathlib import Path
from src.audio_service import AudioService
from src.data_serializer import DataSerializer
import time
import logging
from fp_orchestrator_utils import OrchestratorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize audio service
    audio_service = AudioService()
    data_serializer = DataSerializer()
    orchestrator_client = OrchestratorClient(
        "localhost:50051",
        30
    )
    # Create output directory if it doesn't exist
    output_dir = Path("data", "recorded_audio")
    output_dir.mkdir(exist_ok=True)
    
    try:
        iteration = 1
        while True:
            print(f"\nIteration {iteration}")

            try:  
                # Process audio
                mel_spectrogram = audio_service.StartAudioProcessing()

                # Prepare gRPC payload
                preprocessing_params = {
                    "target_sample_rate": 16000,
                    "target_length": 4,
                    "normalize": True,
                    "normalization_method": "z_score",
                    "trim_strategy": "start",
                }

                feature_params = {
                    "n_fft": 1024,
                    "hop_length": 512,
                    "n_mels": 64,
                    "f_min": 0,
                    "f_max": 8000,
                    "target_sample_rate": 16000,
                    "power": 2.0,
                }

                payload = data_serializer.create_audio_payload(
                    session_id=f"session_{iteration}_{int(time.time())}",
                    mel_spectrogram=mel_spectrogram,
                    preprocessing_params=preprocessing_params,
                    feature_params=feature_params,
                )

                # Verify payload integrity
                if data_serializer._verify_payload(payload):
                    logger.info("Payload verified successfully")
                    # Send payload to orchestrator
                    response = orchestrator_client.send_audio_data(payload)
                    if response.get('status') == 'success':
                        logger.info("Audio data sent successfully")
                else:
                    logger.error("Payload verification failed")

                logger.info(f"Audio processing completed for iteration {iteration}")
                iteration += 1
            except Exception as e:
                logger.error(f"Error during audio processing: {str(e)}")
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")

if __name__ == '__main__':
    main()
