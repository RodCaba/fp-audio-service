from src.grpc_generated import audio_service_pb2, audio_service_pb2_grpc
from src.predictor.predict import AudioPredictor


import pyaudio
from playsound3 import playsound
from gtts import gTTS


import os
import threading
import time
import uuid
import wave
from pathlib import Path


class AudioService(audio_service_pb2_grpc.AudioServiceServicer):
    def __init__(self):
        # Initialize the predictor
        model_path = os.path.join(
                                  "exported_models", "model.onnx")
        print(f"Loading model from: {model_path}")
        self.predictor = AudioPredictor(model_path, feature_type="melspectrogram")

        # Create output directory if it doesn't exist
        self.output_dir = Path("data", "recorded_audio")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Session management
        self.active_sessions = {}
        self.session_lock = threading.Lock()

        print(f"Audio service initialized with model: {model_path}")

        # List available audio devices for debugging
        self._list_audio_devices()

    def StartAudioProcessing(self, request, context):
        """Start audio recording and processing"""
        session_id = request.session_id or str(uuid.uuid4())

        try:
            with self.session_lock:
                if session_id in self.active_sessions:
                    return audio_service_pb2.AudioResponse(
                        session_id=session_id,
                        success=False,
                        error_message="Session already active"
                    )

                # Mark session as active
                self.active_sessions[session_id] = {
                    'status': 'recording',
                    'start_time': time.time()
                }

            print(f"Starting audio processing for session: {session_id}")

            # Record audio
            audio_file = self.output_dir / f"audio_{session_id}.wav"
            recording_duration = request.recording_duration or 5

            self._update_session_status(session_id, 'recording')
            self._record_audio(str(audio_file), seconds=recording_duration)

            # Process audio
            self._update_session_status(session_id, 'processing')
            predicted_class, confidence, all_probabilities = self.predictor.predict(str(audio_file))

            # Create response with top predictions
            top_predictions = []
            import numpy as np

            if hasattr(self.predictor, 'class_names') and self.predictor.class_names:
                # Get top 3 indices
                top_indices = np.argsort(all_probabilities)[-3:][::-1]
                for idx in top_indices:
                    if idx < len(self.predictor.class_names):
                        class_name = self.predictor.class_names[idx]
                        probability = float(all_probabilities[idx])
                        top_predictions.append(
                            audio_service_pb2.ClassProbability(
                                class_name=class_name,
                                probability=probability
                            )
                        )
            else:
                # Fallback if class_names not available
                top_indices = np.argsort(all_probabilities)[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    top_predictions.append(
                        audio_service_pb2.ClassProbability(
                            class_name=f"class_{idx}",
                            probability=float(all_probabilities[idx])
                        )
                    )

            # Generate and play TTS feedback
            self._generate_tts_feedback(predicted_class, confidence, session_id)

            # Mark session as completed
            self._update_session_status(session_id, 'completed')

            # Clean up session after a delay
            threading.Timer(30.0, self._cleanup_session, args=[session_id]).start()

            return audio_service_pb2.AudioResponse(
                session_id=session_id,
                success=True,
                predicted_class=predicted_class,
                confidence=float(confidence),
                top_predictions=top_predictions
            )

        except Exception as e:
            error_msg = f"Audio processing failed: {str(e)}"
            print(f"Error in session {session_id}: {error_msg}")

            self._update_session_status(session_id, 'error')

            return audio_service_pb2.AudioResponse(
                session_id=session_id,
                success=False,
                error_message=error_msg
            )

    def GetProcessingStatus(self, request, context):
        """Get the status of audio processing"""
        session_id = request.session_id

        with self.session_lock:
            if session_id not in self.active_sessions:
                return audio_service_pb2.StatusResponse(
                    session_id=session_id,
                    status="not_found",
                    current_operation="Session not found"
                )

            session_data = self.active_sessions[session_id]
            return audio_service_pb2.StatusResponse(
                session_id=session_id,
                status=session_data['status'],
                current_operation=f"Session started at {session_data['start_time']}"
            )

    def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            # Simple health check - verify predictor is loaded
            if hasattr(self, 'predictor') and self.predictor:
                return audio_service_pb2.HealthCheckResponse(
                    status="SERVING",
                    message="Audio service is healthy"
                )
            else:
                return audio_service_pb2.HealthCheckResponse(
                    status="NOT_SERVING",
                    message="Predictor not loaded"
                )
        except Exception as e:
            return audio_service_pb2.HealthCheckResponse(
                status="NOT_SERVING",
                message=f"Health check failed: {str(e)}"
            )

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

    def _generate_tts_feedback(self, predicted_class, confidence, session_id):
        """Generate and play TTS feedback"""
        try:
            text_to_speak = f"Prediction: {predicted_class}; with confidence {confidence:.2f}"
            tts = gTTS(text=text_to_speak, lang='en')
            tts_file = self.output_dir / f"prediction_{session_id}.mp3"
            tts.save(str(tts_file))
            print(f"Playing back prediction audio: {tts_file}")
            os.system("vlc --play-and-exit " + str(tts_file))  # Use VLC for playback
            # Remove the TTS file after playback
            threading.Timer(10.0, os.remove, args=[str(tts_file)]).start()
        except Exception as e:
            print(f"TTS feedback failed: {e}")

    def _update_session_status(self, session_id, status):
        """Update session status"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = status

    def _cleanup_session(self, session_id):
        """Clean up completed session"""
        with self.session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                print(f"Cleaned up session: {session_id}")

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