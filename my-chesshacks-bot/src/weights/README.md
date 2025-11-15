# Model Weights Directory

This directory stores neural network model weights for the knight0 chess engine.

## Supported Formats

The engine auto-detects and loads models in the following formats (in priority order):

1. **ONNX** (`.onnx`) - **Recommended for production**
   - Optimized for CPU inference
   - Fast startup time
   - Cross-platform compatibility
   - No PyTorch dependency required

2. **TorchScript** (`.pt`, `.pth`)
   - PyTorch serialized models
   - Requires `torch` package installed
   - Good for rapid prototyping

## Model Requirements

Your model must output:
- **Policy**: `(batch_size, 4096)` - Move probabilities (64×64 from-to encoding)
- **Value**: `(batch_size, 1)` or `(batch_size,)` - Position evaluation [-1, 1]

Input format:
- **Shape**: `(batch_size, 21, 8, 8)`
- **Channels**: See `src/board_encoder.py` for channel definitions
- **Type**: `float32`

## Adding Your Model

### Option 1: Local Model (Small Models)

Place your model file in this directory:

```bash
# For ONNX
cp /path/to/your/model.onnx ./
# For PyTorch
cp /path/to/your/model.pt ./
```

The engine will automatically detect and load it on startup.

### Option 2: HuggingFace Hub (Larger Models)

For models > 100MB, host on HuggingFace and load programmatically:

```python
# In src/main.py, replace get_model() initialization:
from .model import load_model_from_huggingface

model = load_model_from_huggingface(
    repo_id="your-username/knight0-model",
    filename="model.onnx"
)
```

### Option 3: Remote URL

Download during initialization:

```python
import urllib.request
from pathlib import Path

model_url = "https://github.com/your-repo/releases/download/v1.0/model.onnx"
weights_dir = Path(__file__).parent / "weights"
model_path = weights_dir / "model.onnx"

if not model_path.exists():
    urllib.request.urlretrieve(model_url, model_path)
```

## Model Export Examples

### Exporting PyTorch to ONNX

```python
import torch
import torch.onnx

# Assuming your model is already trained
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 21, 8, 8)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }
)
```

### Exporting to TorchScript

```python
import torch

model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

## Testing Your Model

Test your model locally before deploying:

```bash
# Run the encoder test
python -m src.board_encoder

# Run the model test
python -m src.model

# Run the full integration
cd devtools && npm run dev
```

## Model Size Recommendations

- **Knight's Edge Category**: < 10MB (lightweight)
- **Queen's Crown Category**: < 100MB (strong)
- **General Competition**: < 500MB (maximum)

Larger models = slower inference = less time for search/lookahead.

## Troubleshooting

### Model Not Loading

Check logs in the Next.js terminal when running `npm run dev`:
```
[knight0] INFO: Loading model from: /path/to/model.onnx
[knight0] INFO: ✓ Model loaded: onnx
```

### Wrong Output Shape

Ensure your model outputs match the expected format:
- Policy must be `(batch, 4096)` 
- Value must be `(batch, 1)` or `(batch,)`

### Slow Inference

- Use ONNX instead of PyTorch
- Quantize your model (int8/float16)
- Reduce model size (fewer layers/channels)
- Disable lookahead search in `src/main.py`

## Current Model

**Status**: No model loaded (using dummy random policy)

To add your trained model, place it in this directory and restart the dev server.

