import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataSerializer:
    """
    Class to serialize audio data to be sent over gRPC.
    """
    def __init__(self):
        pass
    
    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """
        Convert a tensor to bytes for gRPC transmission.
        """
        np_array = tensor.detach().cpu().numpy()
        return np_array.tobytes()

    def _bytes_to_tensor(self, data: bytes, shape: tuple, dtype: str) -> torch.Tensor:
        """
        Convert bytes back to a tensor.
        """
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int16": torch.int16,
            "int32": torch.int32,
        }
        numpy_dtype = dtype_map.get(str(dtype), np.float32)
        np_array = np.frombuffer(data, dtype=numpy_dtype).reshape(shape)
        return torch.tensor(np_array)
    
    def create_audio_payload(
        self,
        session_id: str,
        mel_spectrogram: torch.Tensor,
        preprocessing_params: dict,
        feature_params: dict,
    ) -> dict:
        """
        Create an audio payload dictionary to be sent over gRPC.
        """
        feature_data = self._tensor_to_bytes(mel_spectrogram)
        feature_shape = list(mel_spectrogram.shape)

        payload = {
            "session_id": session_id,
            "sample_rate": preprocessing_params.get("sample_rate", 16000),
            "channels": preprocessing_params.get("channels", 1),
            "features": {
                "feature_type": "mel_spectrogram",
                "feature_shape": feature_shape,
                "feature_data": feature_data,
                "data_type": str(mel_spectrogram.dtype),
                "feature_parameters": {
                    "n_fft": feature_params.get("n_fft", 1024),
                    "hop_length": feature_params.get("hop_length", 512),
                    "n_mels": feature_params.get("n_mels", 128),
                    "f_min": feature_params.get("f_min", 0.0),
                    "f_max": feature_params.get("f_max", 8000.0),
                    "target_sample_rate": feature_params.get("target_sample_rate", 16000),
                    "power": feature_params.get("power", 2.0),
                },
            },
            "parameters": {
                "target_sample_rate": preprocessing_params.get("target_sample_rate", 16000),
                "target_length": preprocessing_params.get("target_length", 5),
                "normalize": preprocessing_params.get("normalize", True),
                "normalization_method": preprocessing_params.get("normalization_method", "z_score"),
                "trim_strategy": preprocessing_params.get("trim_strategy", "none"),
            },
        }

        return payload

    def _verify_payload(self, payload: dict) -> bool:
        """
        Verify the integrity of the audio payload.
        """
        required_keys = [
            "session_id",  "features", 
        ]
        for key in required_keys:
            if key not in payload:
                logger.error(f"Missing required key in payload: {key}")
                return False
        
        # Verify features structure
        features = payload.get("features", {})
        feature_data = features.get("feature_data")
        feature_shape = features.get("feature_shape")
        data_type = features.get("data_type", "float32")
        
        # Reconstruct the tensor from the feature data
        reconstructed_tensor = self._bytes_to_tensor(
            feature_data, 
            shape=tuple(feature_shape), 
            dtype=data_type
        )

        expected_size = np.prod(feature_shape)
        if reconstructed_tensor.numel() != expected_size:
            logger.error("Reconstructed tensor size does not match expected size.")
            return False
        
        logger.info("Payload verification successful.")
      
        return True