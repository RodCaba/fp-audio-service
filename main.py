from pathlib import Path
from src.audio_service import AudioService
from src.data_serializer import DataSerializer
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize audio service
    audio_service = AudioService()
    data_serializer = DataSerializer()
    # Create output directory if it doesn't exist
    output_dir = Path("data", "recorded_audio")
    output_dir.mkdir(exist_ok=True)
    
    try:
        iteration = 1
        while True:
            print(f"\nIteration {iteration}")

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
            else:
                logger.error("Payload verification failed")
                
            logger.info(f"Audio processing completed for iteration {iteration}")
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")

if __name__ == '__main__':
    main()
