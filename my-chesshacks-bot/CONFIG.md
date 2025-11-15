# knight0 Configuration Guide

> Fine-tune your engine for maximum performance

---

## 📁 Configuration Files

### Main Configuration: `src/main.py`

All runtime configuration is at the top of `src/main.py`:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================

# Move selection strategy
MOVE_SELECTION_STRATEGY = "greedy"  # "greedy", "sample", "top_k"
TEMPERATURE = 1.0                    # For "sample" mode
TOP_K = 5                            # For "top_k" mode

# Lookahead search
ENABLE_LOOKAHEAD = False             # Enable shallow search
LOOKAHEAD_DEPTH = 1                  # Ply depth (1-3)
LOOKAHEAD_TOP_N = 3                  # Evaluate top N moves
```

---

## 🎮 Move Selection Strategies

### 1. Greedy (Recommended for Competition)

```python
MOVE_SELECTION_STRATEGY = "greedy"
```

**Behavior**: Always picks the move with highest policy probability

**Pros**:
- Strongest play
- Deterministic (reproducible)
- Fast

**Cons**:
- No exploration
- May get stuck in patterns

**Use When**: You want maximum strength

---

### 2. Temperature Sampling

```python
MOVE_SELECTION_STRATEGY = "sample"
TEMPERATURE = 1.0
```

**Behavior**: Samples moves from policy distribution with temperature scaling

**Temperature Effects**:
- `T = 0.1`: Nearly deterministic (like greedy)
- `T = 1.0`: Standard sampling
- `T = 2.0`: More random/exploratory

**Pros**:
- Variety in play style
- Can discover unexpected tactics
- Good for training data generation

**Cons**:
- Non-deterministic
- May play suboptimal moves
- Slightly slower

**Use When**: Testing different lines or generating diverse games

---

### 3. Top-K Sampling

```python
MOVE_SELECTION_STRATEGY = "top_k"
TOP_K = 5
```

**Behavior**: Samples from the top K moves by policy probability

**K Value Effects**:
- `K = 1`: Same as greedy
- `K = 3-5`: Balanced exploration
- `K = 10+`: High variety

**Pros**:
- Balanced between greedy and sampling
- Avoids very bad moves
- Still has variety

**Cons**:
- Non-deterministic
- Requires tuning K

**Use When**: You want strong but varied play

---

## 🔍 Lookahead Search

### Basic Configuration

```python
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 1
LOOKAHEAD_TOP_N = 3
```

### How It Works

1. Get top N moves from policy
2. For each move:
   - Make the move on the board
   - Run inference on new position
   - Get value estimate
   - Negate (opponent's perspective)
3. Choose move leading to best value

### Depth Recommendations

| Depth | Time per Move | Strength Gain | Use Case |
|-------|---------------|---------------|----------|
| 0 (off) | 20ms | Baseline | Fast play |
| 1 | 60ms | +100 ELO | Balanced |
| 2 | 200ms | +200 ELO | Strong play |
| 3 | 600ms | +250 ELO | Maximum |

### Top-N Recommendations

| Top-N | Time Multiplier | Accuracy |
|-------|-----------------|----------|
| 1 | 1× | Low |
| 3 | 3× | Good |
| 5 | 5× | Better |
| 10 | 10× | Best |

### Example Configurations

**Fast Mode (Knight's Edge)**:
```python
ENABLE_LOOKAHEAD = False
```
- Inference: ~20ms
- Strength: Good
- Total time: 40 moves × 20ms = 800ms

**Balanced Mode**:
```python
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 1
LOOKAHEAD_TOP_N = 3
```
- Inference: ~60ms
- Strength: Very Good
- Total time: 40 moves × 60ms = 2,400ms (4% of time budget)

**Strong Mode (Queen's Crown)**:
```python
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 2
LOOKAHEAD_TOP_N = 5
```
- Inference: ~200ms
- Strength: Strong
- Total time: 40 moves × 200ms = 8,000ms (1.3% of time budget)

---

## ⚙️ Model Configuration

### Board Encoder Settings

In `src/board_encoder.py`:

```python
encoder = BoardEncoder(normalize=True)
```

**Options**:
- `normalize=True`: Halfmove and fullmove normalized to [0, 1]
- `normalize=False`: Raw values (may exceed 1.0)

**Recommendation**: Keep `True` unless your model was trained differently

---

### Model Loading

#### Option 1: Auto-detect (Default)

```python
model = get_model()
```
- Searches `src/weights/` for `.onnx`, `.pt`, `.pth`
- Loads first found

#### Option 2: Specific File

```python
from src.model import ChessModel
model = ChessModel(model_path="src/weights/my_model.onnx")
```

#### Option 3: HuggingFace

```python
from src.model import load_model_from_huggingface
model = load_model_from_huggingface(
    repo_id="your-username/knight0",
    filename="model.onnx"
)
```

---

## 🎯 Performance Tuning

### Inference Speed Optimization

#### 1. Use ONNX Runtime

```bash
pip install onnxruntime
```
- 2-5× faster than PyTorch
- Smaller memory footprint

#### 2. Model Quantization

Convert to int8 (ONNX):
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QInt8
)
```
- 2-4× smaller
- 1.5-2× faster
- Minimal accuracy loss

#### 3. Reduce Model Size

```python
# Train smaller model
model = ChessNet(filters=64, num_blocks=5)  # Instead of 128/10
```

#### 4. Disable Lookahead

```python
ENABLE_LOOKAHEAD = False
```

---

## 🔊 Logging Configuration

### Current Setup (in `src/main.py`)

```python
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='[knight0] %(levelname)s: %(message)s'
)
```

### Log Levels

**INFO** (default):
```
[knight0] INFO: Move 1 | Time left: 600000ms
[knight0] INFO: Position value: 0.245
[knight0] INFO: Top 3 moves: ...
```

**DEBUG** (verbose):
```python
logging.basicConfig(level=logging.DEBUG, ...)
```
- Shows tensor shapes
- Timing breakdowns
- Model loading details

**WARNING** (quiet):
```python
logging.basicConfig(level=logging.WARNING, ...)
```
- Only errors and warnings
- Faster (less I/O)

---

## 🎭 Competition Profiles

### Knight's Edge (< 10MB)

**Goal**: Fast, lightweight

```python
# src/main.py
MOVE_SELECTION_STRATEGY = "greedy"
ENABLE_LOOKAHEAD = False

# Model training
ChessNet(filters=64, num_blocks=5)
```

**Expected**:
- Size: ~5MB
- Inference: 10-20ms
- Strength: 1800-2000 ELO

---

### Queen's Crown (< 100MB)

**Goal**: Strong performance

```python
# src/main.py
MOVE_SELECTION_STRATEGY = "greedy"
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 1
LOOKAHEAD_TOP_N = 3

# Model training
ChessNet(filters=128, num_blocks=15)
```

**Expected**:
- Size: ~50MB
- Inference: 50-100ms
- Strength: 2100-2300 ELO

---

### Maximum Strength (< 500MB)

**Goal**: Best possible play

```python
# src/main.py
MOVE_SELECTION_STRATEGY = "greedy"
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 2
LOOKAHEAD_TOP_N = 5

# Model training
ChessNet(filters=256, num_blocks=20)
```

**Expected**:
- Size: ~200MB
- Inference: 100-200ms
- Strength: 2300-2500 ELO

---

## 🧪 Testing Configurations

### Benchmarking Script

```python
# benchmark.py
import time
import numpy as np
from src.model import get_model
from src.board_encoder import encode_board_simple
from chess import Board

model = get_model()
board = Board()
board_tensor = encode_board_simple(board)

# Warmup
for _ in range(10):
    model.predict(board_tensor)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    policy, value = model.predict(board_tensor)
    times.append((time.perf_counter() - start) * 1000)

print(f"Mean: {np.mean(times):.2f}ms")
print(f"Median: {np.median(times):.2f}ms")
print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
```

### Configuration Testing Matrix

Test these combinations to find optimal settings:

| Config | Strategy | Lookahead | Depth | Top-N | Target Time |
|--------|----------|-----------|-------|-------|-------------|
| Fast | greedy | No | - | - | 20ms |
| Balanced | greedy | Yes | 1 | 3 | 60ms |
| Strong | greedy | Yes | 1 | 5 | 100ms |
| Maximum | greedy | Yes | 2 | 5 | 200ms |
| Exploratory | sample | No | - | - | 25ms |
| Varied | top_k | Yes | 1 | 3 | 65ms |

---

## 🐛 Debugging Configurations

### Debug Mode

```python
# Add at top of src/main.py
DEBUG = True

if DEBUG:
    logger.setLevel(logging.DEBUG)
    # Print tensor shapes
    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value}")
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run inference
policy, value = model.predict(board_tensor)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

## ✅ Configuration Checklist

Before competition:

- [ ] Strategy set to `greedy` for maximum strength
- [ ] Lookahead configured based on time budget
- [ ] Model file in `src/weights/` and loading correctly
- [ ] Inference time < 100ms per move
- [ ] Logging not too verbose (use INFO or WARNING)
- [ ] Tested with multiple positions
- [ ] Error handling working (try removing model file)
- [ ] Hot reloading tested (edit config, save, check logs)

---

## 🔄 Dynamic Configuration (Advanced)

### Adjust Based on Time Left

```python
def get_move(ctx: GameContext) -> Move:
    time_left = ctx.timeLeft
    
    # Dynamic lookahead based on time
    if time_left > 300000:  # > 5 minutes
        depth = 2
        top_n = 5
    elif time_left > 60000:  # > 1 minute
        depth = 1
        top_n = 3
    else:  # < 1 minute
        depth = 0  # Disable
        top_n = 1
    
    # ... rest of logic
```

### Position-Based Configuration

```python
def get_move(ctx: GameContext) -> Move:
    board = ctx.board
    
    # More search in endgame (fewer pieces)
    piece_count = len(board.piece_map())
    
    if piece_count < 10:  # Endgame
        lookahead_enabled = True
        depth = 2
    else:
        lookahead_enabled = False
        depth = 0
    
    # ... rest of logic
```

---

## 📊 Recommended Defaults

**For most users**:

```python
MOVE_SELECTION_STRATEGY = "greedy"
TEMPERATURE = 1.0
TOP_K = 5
ENABLE_LOOKAHEAD = True
LOOKAHEAD_DEPTH = 1
LOOKAHEAD_TOP_N = 3
```

This provides:
- Strong play (greedy)
- 1-ply lookahead for tactical sharpness
- ~60ms per move
- Good balance of speed and strength

---

## 💡 Pro Tips

1. **Profile before optimizing**: Measure actual bottlenecks
2. **Test on target hardware**: Laptop vs. server has different timings
3. **Monitor time usage**: Track cumulative time per game
4. **A/B test**: Compare configurations in devtools
5. **Keep it simple**: Start with defaults, tune if needed

---

Ready to configure your champion! ⚙️

