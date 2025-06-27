import grpc
from concurrent import futures
import time
import uuid
import os
import threading
from pathlib import Path
import pyaudio
import wave
from gtts import gTTS
from playsound3 import playsound

from src.predictor.predict import AudioPredictor
from src.grpc_generated import audio_service_pb2
from src.grpc_generated import audio_service_pb2_grpc


class AudioServiceImpl(audio_service_pb2_grpc.AudioServiceServicer):
    def __init__(self):
        # Initialize the predictor
        model_path = os.path.join( 
                                  "exported_models", "model.onnx")
        print(f"Loading model from: {model_path}")
        self.predictor = AudioPredictor(model_path, feature_type="mfcc")
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("data", "recorded_audio")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Session management
        self.active_sessions = {}
        self.session_lock = threading.Lock()
        
        print(f"Audio service initialized with model: {model_path}")
    
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
            for class_name, prob in all_probabilities[:3]:  # Top 3
                top_predictions.append(
                    audio_service_pb2.ClassProbability(
                        class_name=class_name,
                        probability=float(prob)
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
    
    def _record_audio(self, output_file, seconds=5, rate=44100, channels=1, chunk=1024):
        """Record audio from microphone and save to output_file"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            frames_per_buffer=chunk)
            
            print(f"Recording for {seconds} seconds...")
            frames = []
            
            for i in range(0, int(rate / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)
            
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
            playsound(str(tts_file))
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


def serve():
    """Start the gRPC server"""
    port = os.environ.get('GRPC_PORT', '50051')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    audio_service_pb2_grpc.add_AudioServiceServicer_to_server(
        AudioServiceImpl(), server
    )
    
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
