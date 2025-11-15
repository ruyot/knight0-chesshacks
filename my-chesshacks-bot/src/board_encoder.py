"""
Board Encoder Module
====================
Converts chess.Board objects into neural network input tensors.

Architecture:
- 8x8 spatial grid
- Multiple channels representing different piece types and board features
- Efficient numpy-based encoding
"""

import numpy as np
from chess import Board, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from typing import Optional

# Channel indices for the encoding
# We use 12 channels for pieces (6 types × 2 colors) + additional feature channels
PIECE_TO_CHANNEL = {
    (PAWN, True): 0,      # White Pawn
    (KNIGHT, True): 1,    # White Knight
    (BISHOP, True): 2,    # White Bishop
    (ROOK, True): 3,      # White Rook
    (QUEEN, True): 4,     # White Queen
    (KING, True): 5,      # White King
    (PAWN, False): 6,     # Black Pawn
    (KNIGHT, False): 7,   # Black Knight
    (BISHOP, False): 8,   # Black Bishop
    (ROOK, False): 9,     # Black Rook
    (QUEEN, False): 10,   # Black Queen
    (KING, False): 11,    # Black King
}

# Additional feature channels (indices 12+)
CHANNEL_REPETITION = 12      # Threefold repetition counter
CHANNEL_EN_PASSANT = 13      # En passant target square
CHANNEL_CASTLING_WK = 14     # White kingside castling
CHANNEL_CASTLING_WQ = 15     # White queenside castling
CHANNEL_CASTLING_BK = 16     # Black kingside castling
CHANNEL_CASTLING_BQ = 17     # Black queenside castling
CHANNEL_HALFMOVE = 18        # Halfmove clock (normalized)
CHANNEL_FULLMOVE = 19        # Fullmove number (normalized)
CHANNEL_SIDE_TO_MOVE = 20    # 1 if white to move, 0 if black

TOTAL_CHANNELS = 21


class BoardEncoder:
    """
    Encodes chess positions into tensor format for neural network input.
    
    Format: (CHANNELS, 8, 8) numpy array
    - First 12 channels: piece positions (one-hot encoded)
    - Remaining channels: game state features
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize halfmove and fullmove counters
        """
        self.normalize = normalize
        
    def encode(self, board: Board) -> np.ndarray:
        """
        Encode a board position into a tensor.
        
        Args:
            board: python-chess Board object
            
        Returns:
            np.ndarray of shape (TOTAL_CHANNELS, 8, 8) with dtype float32
        """
        # Initialize tensor with zeros
        tensor = np.zeros((TOTAL_CHANNELS, 8, 8), dtype=np.float32)
        
        # Encode piece positions (channels 0-11)
        for square in range(64):
            piece = board.piece_at(square)
            if piece is not None:
                channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
                rank = square // 8
                file = square % 8
                tensor[channel, rank, file] = 1.0
        
        # Encode en passant target (channel 13)
        if board.ep_square is not None:
            ep_rank = board.ep_square // 8
            ep_file = board.ep_square % 8
            tensor[CHANNEL_EN_PASSANT, ep_rank, ep_file] = 1.0
        
        # Encode castling rights (channels 14-17)
        tensor[CHANNEL_CASTLING_WK, :, :] = float(board.has_kingside_castling_rights(True))
        tensor[CHANNEL_CASTLING_WQ, :, :] = float(board.has_queenside_castling_rights(True))
        tensor[CHANNEL_CASTLING_BK, :, :] = float(board.has_kingside_castling_rights(False))
        tensor[CHANNEL_CASTLING_BQ, :, :] = float(board.has_queenside_castling_rights(False))
        
        # Encode halfmove clock (channel 18) - normalized to [0, 1]
        halfmove_normalized = board.halfmove_clock / 100.0 if self.normalize else board.halfmove_clock
        tensor[CHANNEL_HALFMOVE, :, :] = min(halfmove_normalized, 1.0)
        
        # Encode fullmove number (channel 19) - normalized to [0, 1]
        fullmove_normalized = board.fullmove_number / 100.0 if self.normalize else board.fullmove_number
        tensor[CHANNEL_FULLMOVE, :, :] = min(fullmove_normalized, 1.0)
        
        # Encode side to move (channel 20)
        tensor[CHANNEL_SIDE_TO_MOVE, :, :] = float(board.turn)  # True=1.0 (white), False=0.0 (black)
        
        # Encode repetition counter (channel 12) - can be 0, 1, or 2
        # Note: python-chess can_claim_threefold_repetition() checks if >= 3 repetitions
        # For simplicity, we encode a boolean flag here
        tensor[CHANNEL_REPETITION, :, :] = float(board.can_claim_threefold_repetition())
        
        return tensor
    
    def encode_batch(self, boards: list[Board]) -> np.ndarray:
        """
        Encode multiple boards at once.
        
        Args:
            boards: List of Board objects
            
        Returns:
            np.ndarray of shape (batch_size, TOTAL_CHANNELS, 8, 8)
        """
        return np.stack([self.encode(board) for board in boards], axis=0)


def create_mirrored_encoding(tensor: np.ndarray) -> np.ndarray:
    """
    Create a horizontally mirrored version of the board encoding.
    Useful for data augmentation during training.
    
    Args:
        tensor: Encoded board tensor of shape (CHANNELS, 8, 8)
        
    Returns:
        Mirrored tensor
    """
    mirrored = np.flip(tensor, axis=2).copy()  # Flip along file axis
    return mirrored


def encode_board_simple(board: Board) -> np.ndarray:
    """
    Convenience function for quick encoding with default settings.
    
    Args:
        board: python-chess Board object
        
    Returns:
        Encoded tensor ready for inference
    """
    encoder = BoardEncoder()
    return encoder.encode(board)


if __name__ == "__main__":
    # Test the encoder
    from chess import Board
    
    board = Board()
    encoder = BoardEncoder()
    
    # Test initial position
    tensor = encoder.encode(board)
    print(f"Encoded tensor shape: {tensor.shape}")
    print(f"Non-zero elements: {np.count_nonzero(tensor)}")
    
    # Test after some moves
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    
    tensor = encoder.encode(board)
    print(f"\nAfter moves - shape: {tensor.shape}")
    print(f"Side to move (white=1, black=0): {tensor[CHANNEL_SIDE_TO_MOVE, 0, 0]}")
    
    # Verify piece encoding
    print(f"\nWhite pieces (channels 0-5) sum: {tensor[0:6].sum()}")
    print(f"Black pieces (channels 6-11) sum: {tensor[6:12].sum()}")

