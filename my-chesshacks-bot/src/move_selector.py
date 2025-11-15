"""
Move Selection Module
=====================
Converts neural network policy outputs into legal chess moves.

Features:
- Legal move masking
- Multiple selection strategies (greedy, temperature sampling, top-k)
- Move probability logging
- UCI format conversion
"""

import numpy as np
from chess import Board, Move
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def square_to_index(square: int) -> Tuple[int, int]:
    """Convert chess square index (0-63) to rank, file."""
    return square // 8, square % 8


def move_to_policy_index(move: Move) -> int:
    """
    Convert a chess.Move to a policy index (0-4095).
    
    Policy is organized as a 64x64 matrix (flattened):
    - from_square (0-63) × 64 + to_square (0-63)
    
    Args:
        move: chess.Move object
        
    Returns:
        Policy index (0-4095)
    """
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square


def policy_index_to_move(index: int) -> Tuple[int, int]:
    """
    Convert policy index to (from_square, to_square).
    
    Args:
        index: Policy index (0-4095)
        
    Returns:
        Tuple of (from_square, to_square)
    """
    from_square = index // 64
    to_square = index % 64
    return from_square, to_square


def create_legal_move_mask(board: Board, policy_shape: int = 4096) -> np.ndarray:
    """
    Create a binary mask for legal moves.
    
    Args:
        board: chess.Board object
        policy_shape: Size of policy vector (default 4096 for 64x64)
        
    Returns:
        Binary mask array where 1 = legal move, 0 = illegal
    """
    mask = np.zeros(policy_shape, dtype=np.float32)
    
    for move in board.legal_moves:
        policy_idx = move_to_policy_index(move)
        mask[policy_idx] = 1.0
    
    return mask


def mask_illegal_moves(policy: np.ndarray, legal_moves: List[Move]) -> np.ndarray:
    """
    Mask out illegal moves from policy and renormalize.
    
    Args:
        policy: Raw policy output from neural network (4096,)
        legal_moves: List of legal chess.Move objects
        
    Returns:
        Masked and renormalized policy
    """
    masked_policy = np.zeros_like(policy)
    
    # Copy probabilities for legal moves
    for move in legal_moves:
        idx = move_to_policy_index(move)
        masked_policy[idx] = policy[idx]
    
    # Renormalize to sum to 1
    policy_sum = masked_policy.sum()
    if policy_sum > 0:
        masked_policy = masked_policy / policy_sum
    else:
        # If all legal moves have 0 probability, use uniform distribution
        logger.warning("All legal moves have 0 probability. Using uniform distribution.")
        for move in legal_moves:
            idx = move_to_policy_index(move)
            masked_policy[idx] = 1.0 / len(legal_moves)
    
    return masked_policy


class MoveSelector:
    """
    Handles move selection from neural network policy outputs.
    """
    
    def __init__(self, strategy: str = "greedy", temperature: float = 1.0, top_k: Optional[int] = None):
        """
        Args:
            strategy: Selection strategy - "greedy", "sample", or "top_k"
            temperature: Temperature for sampling (higher = more random)
            top_k: If set, only sample from top K moves
        """
        self.strategy = strategy
        self.temperature = temperature
        self.top_k = top_k
    
    def select_move(
        self,
        board: Board,
        policy: np.ndarray,
        value: Optional[float] = None
    ) -> Tuple[Move, Dict[Move, float]]:
        """
        Select a move from the policy.
        
        Args:
            board: Current board position
            policy: Policy output from neural network (4096,)
            value: Optional value estimate (not used in basic selection)
            
        Returns:
            Tuple of:
                - Selected Move object
                - Dictionary of {Move: probability} for all legal moves
        """
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Apply legal move mask
        masked_policy = mask_illegal_moves(policy, legal_moves)
        
        # Extract probabilities for legal moves
        move_probs = {}
        for move in legal_moves:
            idx = move_to_policy_index(move)
            move_probs[move] = float(masked_policy[idx])
        
        # Select move based on strategy
        if self.strategy == "greedy":
            selected_move = self._select_greedy(move_probs)
        elif self.strategy == "sample":
            selected_move = self._select_sample(move_probs)
        elif self.strategy == "top_k":
            selected_move = self._select_top_k(move_probs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return selected_move, move_probs
    
    def _select_greedy(self, move_probs: Dict[Move, float]) -> Move:
        """Select move with highest probability."""
        return max(move_probs.items(), key=lambda x: x[1])[0]
    
    def _select_sample(self, move_probs: Dict[Move, float]) -> Move:
        """Sample move from probability distribution with temperature."""
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])
        
        # Apply temperature
        if self.temperature != 1.0:
            # Convert to logits, apply temperature, then softmax
            logits = np.log(probs + 1e-8)
            logits = logits / self.temperature
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
        
        # Sample
        selected_idx = np.random.choice(len(moves), p=probs)
        return moves[selected_idx]
    
    def _select_top_k(self, move_probs: Dict[Move, float]) -> Move:
        """Sample from top K moves."""
        # Sort moves by probability
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Take top K
        k = self.top_k if self.top_k else len(sorted_moves)
        top_moves = sorted_moves[:k]
        
        # Renormalize and sample
        moves = [m for m, _ in top_moves]
        probs = np.array([p for _, p in top_moves])
        probs = probs / probs.sum()
        
        selected_idx = np.random.choice(len(moves), p=probs)
        return moves[selected_idx]


def select_best_move_simple(board: Board, policy: np.ndarray) -> Tuple[Move, Dict[Move, float]]:
    """
    Simple greedy move selection (convenience function).
    
    Args:
        board: Current chess position
        policy: Neural network policy output
        
    Returns:
        Tuple of (selected_move, move_probabilities_dict)
    """
    selector = MoveSelector(strategy="greedy")
    return selector.select_move(board, policy)


def select_move_with_lookahead(
    board: Board,
    policy: np.ndarray,
    value: float,
    model,
    top_n: int = 5,
    depth: int = 1
) -> Tuple[Move, Dict[Move, float]]:
    """
    Advanced move selection with shallow lookahead search.
    
    This evaluates the top N moves by looking ahead one or more plies
    and choosing the move that leads to the best position according to
    the value head.
    
    Args:
        board: Current position
        policy: Policy from neural network
        value: Value estimate for current position
        model: Neural network model for evaluating future positions
        top_n: Number of top moves to consider
        depth: Lookahead depth (1 = one ply)
        
    Returns:
        Tuple of (selected_move, move_probabilities_dict)
    """
    from .board_encoder import encode_board_simple
    
    legal_moves = list(board.legal_moves)
    masked_policy = mask_illegal_moves(policy, legal_moves)
    
    # Get top N moves by policy
    move_probs = {}
    for move in legal_moves:
        idx = move_to_policy_index(move)
        move_probs[move] = float(masked_policy[idx])
    
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    top_moves = sorted_moves[:top_n]
    
    # Evaluate each top move
    move_values = {}
    for move, prob in top_moves:
        # Make move
        board.push(move)
        
        # Encode and evaluate
        board_tensor = encode_board_simple(board)
        _, future_value = model.predict(board_tensor)
        
        # Negate value (opponent's perspective)
        move_values[move] = -float(future_value)
        
        # Undo move
        board.pop()
    
    # Select move with best value
    best_move = max(move_values.items(), key=lambda x: x[1])[0]
    
    logger.info(f"Lookahead values: {[(m.uci(), v) for m, v in move_values.items()]}")
    
    return best_move, move_probs


if __name__ == "__main__":
    # Test move selector
    from chess import Board
    
    board = Board()
    
    # Create dummy policy
    policy = np.random.rand(4096).astype(np.float32)
    policy = policy / policy.sum()
    
    # Test greedy selection
    selector = MoveSelector(strategy="greedy")
    move, probs = selector.select_move(board, policy)
    
    print(f"Selected move: {move.uci()}")
    print(f"Number of legal moves: {len(probs)}")
    print(f"Top 5 moves by probability:")
    for move, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {move.uci()}: {prob:.4f}")

