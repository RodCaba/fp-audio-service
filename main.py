from pathlib import Path
from src.audio_service import AudioService


def main():
    # Initialize audio service
    audio_service = AudioService()    
    # Create output directory if it doesn't exist
    output_dir = Path("data", "recorded_audio")
    output_dir.mkdir(exist_ok=True)
    
    try:
        iteration = 1
        while True:
            print(f"\nIteration {iteration}")

            # Process audio
            mel_spectrogram = audio_service.StartAudioProcessing()
            print(f"Audio processing completed for iteration {iteration}")
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")

if __name__ == '__main__':
    main()
