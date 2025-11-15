"""
Neural Network Model Module
============================
Handles loading and inference for chess neural networks.

Supports:
- ONNX models (onnxruntime - CPU optimized)
- TorchScript models (PyTorch)
- Direct PyTorch models

Model Output Format:
- policy: (batch_size, 4096) - move probabilities (64 from × 64 to squares)
- value: (batch_size, 1) - position evaluation [-1, 1]
"""

import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ChessModel:
    """
    Base class for chess model inference.
    Handles both ONNX and PyTorch models.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Args:
            model_path: Path to model file (.onnx, .pt, or .pth)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.model_type = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load model from file. Auto-detects format from extension.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Running in dummy mode with random policy")
            self.model_type = "dummy"
            self.is_loaded = True
            return
        
        extension = model_path.suffix.lower()
        
        if extension == ".onnx":
            self._load_onnx(str(model_path))
        elif extension in [".pt", ".pth"]:
            self._load_pytorch(str(model_path))
        else:
            raise ValueError(f"Unsupported model format: {extension}")
        
        self.model_path = str(model_path)
        self.is_loaded = True
        logger.info(f"Successfully loaded {self.model_type} model from {model_path}")
    
    def _load_onnx(self, model_path: str):
        """Load ONNX model using onnxruntime."""
        try:
            import onnxruntime as ort
            
            # Create inference session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # Adjust based on CPU
            
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.model = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.model_type = "onnx"
            
            # Log input/output info
            input_name = self.model.get_inputs()[0].name
            input_shape = self.model.get_inputs()[0].shape
            logger.info(f"ONNX model input: {input_name} {input_shape}")
            
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_pytorch(self, model_path: str):
        """Load PyTorch or TorchScript model."""
        try:
            import torch
            
            # Try loading as TorchScript first
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                self.model_type = "torchscript"
            except:
                # Fall back to regular PyTorch model
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.model_type = "pytorch"
            
            logger.info(f"Loaded PyTorch model on {self.device}")
            
        except ImportError:
            logger.error("PyTorch not installed. Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def predict(self, board_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on encoded board position(s).
        
        Args:
            board_tensor: Numpy array of shape (CHANNELS, 8, 8) or (batch, CHANNELS, 8, 8)
            
        Returns:
            Tuple of:
                - policy: numpy array of shape (4096,) or (batch, 4096)
                - value: float or numpy array of shape (batch,)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle batching
        if board_tensor.ndim == 3:
            board_tensor = np.expand_dims(board_tensor, axis=0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if self.model_type == "dummy":
            return self._dummy_predict(board_tensor, squeeze_output)
        elif self.model_type == "onnx":
            return self._predict_onnx(board_tensor, squeeze_output)
        elif self.model_type in ["pytorch", "torchscript"]:
            return self._predict_pytorch(board_tensor, squeeze_output)
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")
    
    def _dummy_predict(self, board_tensor: np.ndarray, squeeze: bool) -> Tuple[np.ndarray, float]:
        """Dummy prediction with random policy (for testing without a real model)."""
        batch_size = board_tensor.shape[0]
        
        # Random policy
        policy = np.random.rand(batch_size, 4096).astype(np.float32)
        # Normalize to sum to 1
        policy = policy / policy.sum(axis=1, keepdims=True)
        
        # Random value around 0
        value = np.random.randn(batch_size).astype(np.float32) * 0.3
        
        if squeeze:
            policy = policy[0]
            value = value[0]
        
        return policy, value
    
    def _predict_onnx(self, board_tensor: np.ndarray, squeeze: bool) -> Tuple[np.ndarray, float]:
        """Run ONNX inference."""
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: board_tensor})
        
        # Expecting [policy, value] as outputs
        policy = outputs[0]  # Shape: (batch, 4096)
        value = outputs[1]   # Shape: (batch, 1) or (batch,)
        
        # Flatten value if needed
        if value.ndim == 2:
            value = value.squeeze(-1)
        
        if squeeze:
            policy = policy[0]
            value = value[0]
        
        return policy, value
    
    def _predict_pytorch(self, board_tensor: np.ndarray, squeeze: bool) -> Tuple[np.ndarray, float]:
        """Run PyTorch inference."""
        import torch
        
        # Convert to torch tensor
        tensor = torch.from_numpy(board_tensor).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                policy, value = outputs
            elif isinstance(outputs, dict):
                policy = outputs['policy']
                value = outputs['value']
            else:
                # Assume single output is policy only
                policy = outputs
                value = torch.zeros(board_tensor.shape[0], device=self.device)
        
        # Convert back to numpy
        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
        
        if value.ndim == 2:
            value = value.squeeze(-1)
        
        if squeeze:
            policy = policy[0]
            value = value[0]
        
        return policy, value


# Global model instance (loaded once at module import)
_global_model: Optional[ChessModel] = None


def get_model(force_reload: bool = False) -> ChessModel:
    """
    Get the global model instance (singleton pattern).
    
    Args:
        force_reload: If True, reload the model even if already loaded
        
    Returns:
        ChessModel instance
    """
    global _global_model
    
    if _global_model is None or force_reload:
        # Look for model in weights directory
        weights_dir = Path(__file__).parent / "weights"
        model_path = None
        
        # Search for model files in priority order
        for extension in [".onnx", ".pt", ".pth"]:
            candidates = list(weights_dir.glob(f"*{extension}"))
            if candidates:
                model_path = str(candidates[0])
                break
        
        if model_path:
            logger.info(f"Loading model from: {model_path}")
            _global_model = ChessModel(model_path=model_path)
        else:
            logger.warning("No model file found in weights directory")
            logger.warning("Running with dummy random policy")
            _global_model = ChessModel()  # Will use dummy mode
            _global_model.model_type = "dummy"
            _global_model.is_loaded = True
    
    return _global_model


def load_model_from_huggingface(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> ChessModel:
    """
    Load a model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        filename: Name of the model file in the repo
        cache_dir: Optional cache directory for downloaded models
        
    Returns:
        ChessModel instance
    """
    try:
        from huggingface_hub import hf_hub_download
        
        logger.info(f"Downloading model from HuggingFace: {repo_id}/{filename}")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        
        model = ChessModel(model_path=model_path)
        return model
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        raise


if __name__ == "__main__":
    # Test model loading
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    # Test with dummy model
    model = get_model()
    print(f"Model loaded: {model.model_type}")
    
    # Test inference with random input
    dummy_input = np.random.randn(1, 21, 8, 8).astype(np.float32)
    policy, value = model.predict(dummy_input)
    
    print(f"Policy shape: {policy.shape}, sum: {policy.sum():.4f}")
    print(f"Value: {value:.4f}")
    print(f"Top 5 policy moves: {np.argsort(policy)[-5:]}")

