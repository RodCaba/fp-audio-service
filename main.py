import pyaudio
import wave
import time
import os
from pathlib import Path
from src.predictor.predict import AudioPredictor
import pygame
from gtts import gTTS

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

def play_audio_with_pygame(audio_file):
    """Play audio file using pygame"""
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            # Try different audio drivers for better compatibility
            audio_drivers = ['alsa', 'pulse', 'oss', 'dummy']
            mixer_initialized = False
            
            for driver in audio_drivers:
                try:
                    import os
                    os.environ['SDL_AUDIODRIVER'] = driver
                    pygame.mixer.init()
                    print(f"Audio initialized with {driver} driver")
                    mixer_initialized = True
                    break
                except Exception as e:
                    print(f"Failed to initialize audio with {driver}: {e}")
                    continue
            
            if not mixer_initialized:
                print("No audio driver available - running in silent mode")
                return
        
        # Load and play the audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Pygame audio playback failed: {e}")
    finally:
        # Clean up
        try:
            pygame.mixer.music.stop()
        except:
            pass

def main():
    # Initialize pygame mixer for audio playback
    try:
        pygame.mixer.init()
        print("Pygame mixer initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize pygame mixer: {e}")
    
    # Initialize the predictor
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_models", "model.onnx")
    predictor = AudioPredictor(model_path, feature_type="mfcc")
    
    # Create output directory if it doesn't exist
    output_dir = Path("data", "recorded_audio")
    output_dir.mkdir(exist_ok=True)
    
    try:
        iteration = 1
        rest_time = 2  # Time to rest between recordings in seconds
        
        while True:
            audio_file = output_dir / "audio.wav"
            print(f"\nIteration {iteration}")

            # Record audio
            record_audio(str(audio_file), seconds=5)
            
            # Run prediction
            print("Running prediction...")
            predicted_class, confidence, all_probabilities = predictor.predict(str(audio_file))
            
            # Print results
            print("Prediction results:")
            print(f"Class: {predicted_class}, Confidence: {confidence:.2f}")
            print("Top 3 probabilities:", all_probabilities[:3])

            # Play recorded audio   
            print(f"Playing back recorded audio: {audio_file}")
            play_audio_with_pygame(str(audio_file))

            text_to_speak = f"Prediction: {predicted_class}; with confidence {confidence:.2f}"
            tts = gTTS(text=text_to_speak, lang='en')
            tts_file = output_dir / "prediction.mp3"
            tts.save(str(tts_file))
            print(f"Playing back prediction audio: {tts_file}")
            play_audio_with_pygame(str(tts_file))

            print(f"Resting for {rest_time} seconds before next recording...")
            time.sleep(rest_time)
            
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")

if __name__ == '__main__':
    main()
