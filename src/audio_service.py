import pyaudio
import wave
from pathlib import Path
from .preprocessing.preprocessing import PreprocessingService
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioService():
    def __init__(self):

        # Create output directory if it doesn't exist
        self.output_dir = Path("data", "recorded_audio")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # List available audio devices for debugging
        self._list_audio_devices()
        self.preprocessor = PreprocessingService()

    def StartAudioProcessing(self):
        """Start audio recording and processing"""

        try:
            # Record audio
            audio_file = self.output_dir / f"audio.wav"

            self._record_audio(str(audio_file), seconds=5)

            # Process audio
            mel_spectrogram = self.preprocessor.compute_melspectrogram(str(audio_file))
            logger.info(f"Mel-spectrogram shape: {mel_spectrogram.shape}")
            logger.info(f"Mel-spectrogram mean: {mel_spectrogram.mean()}, std: {mel_spectrogram.std()}")

            return mel_spectrogram

        except Exception as e:
            error_msg = f"Audio processing failed: {str(e)}"
            logger.error(error_msg)

    def _record_audio(self, output_file, seconds=5, rate=44100, channels=1, chunk=4096):
        """Record audio from microphone and save to output_file"""
        p = pyaudio.PyAudio()

        try:
            # Find the USB audio device (UM02)
            input_device_index = None
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if 'UM02' in device_info['name'] or 'USB Audio' in device_info['name']:
                    if device_info['maxInputChannels'] > 0:
                        input_device_index = i
                        print(f"Using audio device: {device_info['name']} (index: {i})")
                        break

            # If no USB device found, use default input device
            if input_device_index is None:
                input_device_index = p.get_default_input_device_info()['index']
                print(f"Using default input device (index: {input_device_index})")

            # Open stream with error handling for buffer overflow
            stream = p.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=chunk,
                            start=False)  # Don't start immediately

            print(f"Recording for {seconds} seconds...")
            frames = []

            # Start the stream
            stream.start_stream()

            # Record with exception handling for overflow
            for i in range(0, int(rate / chunk * seconds)):
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        # Handle input overflow by skipping this chunk
                        print(f"Input overflow detected, skipping chunk {i}")
                        # Create silence for the missed chunk
                        silence = b'\x00' * (chunk * 2)  # 2 bytes per sample for paInt16
                        frames.append(silence)
                    else:
                        raise e

            print("Recording finished")

            stream.stop_stream()
            stream.close()

            # Save as WAV file
            wf = wave.open(output_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"Audio saved to {output_file}")

        except Exception as e:
            print(f"Recording error: {e}")
            # Create a minimal silence file if recording fails completely
            try:
                wf = wave.open(output_file, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(rate)
                # Write 1 second of silence
                silence_frames = int(rate * 1)
                silence_data = b'\x00' * (silence_frames * 2)
                wf.writeframes(silence_data)
                wf.close()
                print(f"Created silence file due to recording error: {output_file}")
            except Exception as fallback_error:
                print(f"Failed to create fallback audio file: {fallback_error}")
                raise e
        finally:
            p.terminate()

    def _list_audio_devices(self):
        """List available audio devices for debugging"""
        p = pyaudio.PyAudio()
        try:
            print("Available audio devices:")
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                device_type = []
                if device_info['maxInputChannels'] > 0:
                    device_type.append('INPUT')
                if device_info['maxOutputChannels'] > 0:
                    device_type.append('OUTPUT')

                print(f"  Device {i}: {device_info['name']} ({'/'.join(device_type)})")
                print(f"    Max input channels: {device_info['maxInputChannels']}")
                print(f"    Max output channels: {device_info['maxOutputChannels']}")
                print(f"    Default sample rate: {device_info['defaultSampleRate']}")
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        finally:
            p.terminate()