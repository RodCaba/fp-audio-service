import pyaudio
import wave
import time
import os
from pathlib import Path
from src.predictor.predict import AudioPredictor
from playsound3 import playsound

def record_audio(output_file, seconds=5, rate=44100, channels=1, chunk=1024):
        """Record audio from microphone and save to output_file"""
        p = pyaudio.PyAudio()

        # List available audio devices
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']} (max input channels: {info['maxInputChannels']})")
        
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk,
                    )
        
        print(f"Recording for {seconds} seconds...")
        frames = []
        
        for i in range(0, int(rate / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
        
        print("Recording finished")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save as WAV file
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Audio saved to {output_file}")

def main():
    # Initialize the predictor
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_models", "model.onnx")
    predictor = AudioPredictor(model_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path("data", "recorded_audio")
    output_dir.mkdir(exist_ok=True)
    
    try:
        iteration = 1
        rest_time = 2  # Time to rest between recordings in seconds
        
        while True:
            audio_file = output_dir / "audio.wav"
            print(f"\nIteration {iteration}")

            # Play the audio file if it exists
            if audio_file.exists():
                print(f"Playing existing audio file: {audio_file}")
                playsound(str(audio_file))
            else:
                print("No existing audio file to play.")

            # Record audio
            record_audio(str(audio_file), seconds=5)
            
            # Run prediction
            print("Running prediction...")
            results = predictor.predict(str(audio_file))
            
            # Print results
            print("Prediction results:")
            for label, probability in results.items():
                print(f"  {label}: {probability:.4f}")
            
            print(f"Resting for {rest_time} seconds before next recording...")
            time.sleep(rest_time)
            
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")

if __name__ == '__main__':
    main()
