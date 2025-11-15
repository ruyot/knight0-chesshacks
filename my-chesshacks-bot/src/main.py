"""
knight0 - Neural Chess Engine
==============================
A competitive neural network-based chess engine for ChessHacks.

Architecture:
- ResNet-style CNN for position evaluation
- Policy head for move selection
- Value head for position evaluation
- Fast inference optimized for competition
"""

from .utils import chess_manager, GameContext
from chess import Move
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='[knight0] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import our neural engine components
from .model import get_model
from .board_encoder import BoardEncoder
from .move_selector import MoveSelector

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model selection strategy: "greedy" (strongest), "sample" (more variety), "top_k" (balanced)
MOVE_SELECTION_STRATEGY = "greedy"

# Temperature for sampling (only used if strategy="sample")
TEMPERATURE = 1.0

# Top-K parameter (only used if strategy="top_k")
TOP_K = 5

# Enable lookahead search for stronger play (slower but better)
ENABLE_LOOKAHEAD = False
LOOKAHEAD_DEPTH = 1
LOOKAHEAD_TOP_N = 3

# ============================================================================
# INITIALIZATION (runs once at module import)
# ============================================================================

logger.info("=" * 60)
logger.info("knight0 Chess Engine - Initializing...")
logger.info("=" * 60)

# Load model (singleton - happens once)
try:
    model = get_model()
    logger.info(f"✓ Model loaded: {model.model_type}")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    logger.warning("Falling back to random play")
    model = None

# Initialize encoder and move selector
encoder = BoardEncoder(normalize=True)
move_selector = MoveSelector(
    strategy=MOVE_SELECTION_STRATEGY,
    temperature=TEMPERATURE,
    top_k=TOP_K
)

logger.info(f"✓ Board encoder initialized")
logger.info(f"✓ Move selector: {MOVE_SELECTION_STRATEGY}")
logger.info(f"✓ Lookahead: {'enabled' if ENABLE_LOOKAHEAD else 'disabled'}")
logger.info("=" * 60)

# ============================================================================
# MOVE GENERATION
# ============================================================================

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    """
    Main entrypoint for move generation.
    
    This function is called every time the bot needs to make a move.
    
    Args:
        ctx: GameContext with current board state and time remaining
        
    Returns:
        chess.Move object representing the chosen move
    """
    board = ctx.board
    time_left = ctx.timeLeft
    
    logger.info(f"Move {len(board.move_stack) + 1} | Time left: {time_left}ms")
    
    # Check for legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    try:
        # Step 1: Encode board position
        board_tensor = encoder.encode(board)
        
        # Step 2: Run neural network inference
        policy, value = model.predict(board_tensor)
        
        logger.info(f"Position value: {value:.3f}")
        
        # Step 3: Select move
        if ENABLE_LOOKAHEAD and model is not None:
            from .move_selector import select_move_with_lookahead
            selected_move, move_probs = select_move_with_lookahead(
                board=board,
                policy=policy,
                value=value,
                model=model,
                top_n=LOOKAHEAD_TOP_N,
                depth=LOOKAHEAD_DEPTH
            )
        else:
            selected_move, move_probs = move_selector.select_move(
                board=board,
                policy=policy,
                value=value
            )
        
        # Log move probabilities for visualization
        ctx.logProbabilities(move_probs)
        
        # Log top 3 moves
        top_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info("Top 3 moves:")
        for i, (move, prob) in enumerate(top_moves, 1):
            logger.info(f"  {i}. {move.uci()} ({prob:.2%})")
        
        logger.info(f"Selected: {selected_move.uci()}\n")
        
        return selected_move
        
    except Exception as e:
        logger.error(f"Error in move generation: {e}")
        logger.warning("Falling back to random legal move")
        
        # Emergency fallback: random move
        import random
        fallback_move = random.choice(legal_moves)
        
        # Create uniform probabilities
        uniform_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(uniform_probs)
        
        return fallback_move


@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Called when a new game begins.
    Use this to clear any caches or reset state.
    """
    logger.info("=" * 60)
    logger.info("New game started - knight0 ready!")
    logger.info("=" * 60)
