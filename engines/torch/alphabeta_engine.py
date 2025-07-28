#!/usr/bin/env python3
import chess
from chess import Board, Move
import time
import torch
import numpy as np
from typing import Tuple, Optional, List
try:
    from predict import ChessPredictor
except ImportError:
    ChessPredictor = None
    print("Warning: Neural predictor not available")


class AlphaBetaEngine:
    """Alpha-Beta engine with optional neural network guidance"""
    
    # Traditional piece values for evaluation
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Position bonuses for pieces (from white's perspective)
    PAWN_TABLE = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    KNIGHT_TABLE = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    BISHOP_TABLE = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    ROOK_TABLE = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [0,  0,  0,  5,  5,  0,  0,  0]
    ]
    
    QUEEN_TABLE = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [-5,  0,  5,  5,  5,  5,  0, -5],
        [0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ]
    
    KING_MIDDLE_TABLE = [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [20, 20,  0,  0,  0,  0, 20, 20],
        [20, 30, 10,  0,  0, 10, 30, 20]
    ]
    
    def __init__(self, max_depth=4, neural_model_path=None, neural_mapping_path=None, 
                 neural_weight=0.4):
        """
        Initialize alpha-beta engine
        
        Args:
            max_depth: Maximum search depth
            neural_model_path: Path to neural network model (optional)
            neural_mapping_path: Path to neural network move mapping (optional)
            neural_weight: Weight for neural evaluation (0.0 = pure traditional, 1.0 = pure neural)
        """
        self.max_depth = max_depth
        self.nodes_searched = 0
        self.neural_weight = neural_weight
        self.neural_predictor = None
        
        # Initialize neural network predictor if available
        print(f"Initializing AlphaBetaEngine with neural_weight={neural_weight}")
        
        if ChessPredictor is None:
            print("✗ ChessPredictor not available - neural integration disabled")
            self.neural_weight = 0.0
            return
        
        if neural_weight <= 0.0:
            print("✓ Neural integration disabled (neural_weight=0.0)")
            return
        
        # Try to initialize neural predictor
        print("Attempting to load neural predictor...")
        
        try:
            if neural_model_path and neural_mapping_path:
                print(f"  Using explicit paths: {neural_model_path}, {neural_mapping_path}")
                self.neural_predictor = ChessPredictor(neural_model_path, neural_mapping_path)
            else:
                print("  Using default paths")
                self.neural_predictor = ChessPredictor()
            
            print(f"✓ Neural guidance enabled (weight: {neural_weight})")
            print(f"  └─ Device: {self.neural_predictor.device}")
            print(f"  └─ Classes: {self.neural_predictor.num_classes}")
            
        except FileNotFoundError as e:
            print(f"✗ Neural model files not found: {e}")
            print("  └─ Falling back to pure traditional evaluation")
            self.neural_predictor = None
            self.neural_weight = 0.0
            
        except Exception as e:
            print(f"✗ Failed to load neural predictor: {e}")
            print("  └─ Falling back to pure traditional evaluation")
            self.neural_predictor = None
            self.neural_weight = 0.0
    
    def get_piece_square_value(self, piece: chess.Piece, square: int) -> int:
        """Get positional value for piece on square"""
        piece_type = piece.piece_type
        color = piece.color
        
        # Convert square to row, col (0-7)
        row = square // 8
        col = square % 8
        
        # Flip row for black pieces
        if not color:  # Black
            row = 7 - row
            
        if piece_type == chess.PAWN:
            return self.PAWN_TABLE[row][col]
        elif piece_type == chess.KNIGHT:
            return self.KNIGHT_TABLE[row][col]
        elif piece_type == chess.BISHOP:
            return self.BISHOP_TABLE[row][col]
        elif piece_type == chess.ROOK:
            return self.ROOK_TABLE[row][col]
        elif piece_type == chess.QUEEN:
            return self.QUEEN_TABLE[row][col]
        elif piece_type == chess.KING:
            return self.KING_MIDDLE_TABLE[row][col]
        
        return 0
    
    def traditional_evaluation(self, board: Board) -> int:
        """
        Traditional board evaluation (material + position + mobility)
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            int: Evaluation score (positive = good for white)
        """
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and positional evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_value = self.PIECE_VALUES[piece.piece_type]
                position_value = self.get_piece_square_value(piece, square)
                
                if piece.color:  # White
                    score += piece_value + position_value
                else:  # Black
                    score -= piece_value + position_value
        
        # Mobility bonus
        legal_moves = len(list(board.legal_moves))
        if board.turn:  # White to move
            score += legal_moves * 2
        else:  # Black to move
            score -= legal_moves * 2
        
        return score
    
    def neural_evaluation(self, board: Board) -> int:
        """
        Neural network based evaluation
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            int: Evaluation score from neural network perspective
        """
        if not self.neural_predictor:
            return 0
        
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        try:
            # Get top moves with probabilities
            top_moves = self.neural_predictor.get_top_moves(board, top_k=5)
            if not top_moves:
                return 0
            
            # Calculate confidence score based on move probabilities
            total_prob = sum(prob for _, prob in top_moves)
            best_move_prob = top_moves[0][1] if top_moves else 0
            
            # Higher confidence in good moves = better position
            confidence_score = int((total_prob * 500) + (best_move_prob * 500))
            
            # Adjust sign for current player
            return confidence_score if board.turn else -confidence_score
            
        except Exception as e:
            # Fallback to traditional evaluation if neural network fails
            print(f"Neural evaluation failed: {e}")
            return 0
    
    def hybrid_evaluation(self, board: Board) -> int:
        """
        Combine traditional and neural evaluations
        
        Args:
            board: Board to evaluate
            
        Returns:
            int: Combined evaluation score
        """
        traditional_eval = self.traditional_evaluation(board)
        
        if self.neural_predictor and self.neural_weight > 0:
            neural_eval = self.neural_evaluation(board)
            # Weighted combination
            combined = ((1 - self.neural_weight) * traditional_eval + 
                       self.neural_weight * neural_eval)
            return int(combined)
        else:
            return traditional_eval
    
    def neural_move_ordering(self, board: Board) -> List[Move]:
        """
        Order moves using neural network predictions + traditional heuristics
        
        Args:
            board: Current board position
            
        Returns:
            list: Ordered list of moves (best first)
        """
        moves = list(board.legal_moves)
        if not moves:
            return moves
        
        if not self.neural_predictor:
            return self.traditional_move_ordering(board)
        
        def move_priority(move):
            priority = 0
            
            # Traditional heuristics (captures, checks, promotions)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    priority += self.PIECE_VALUES[captured_piece.piece_type]
                
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                attacker = board.piece_at(move.from_square)
                if attacker:
                    priority -= self.PIECE_VALUES[attacker.piece_type] // 10
            
            # Check bonus
            board.push(move)
            if board.is_check():
                priority += 50
            board.pop()
            
            # Promotion bonus
            if move.promotion:
                priority += 800
            
            # Neural network guidance
            try:
                move_uci = move.uci()
                if move_uci in self.neural_predictor.move_to_int:
                    # Get neural network probability for this move
                    X_tensor = self.neural_predictor.prepare_input(board).to(self.neural_predictor.device)
                    with torch.no_grad():
                        logits = self.neural_predictor.model(X_tensor)
                    probabilities = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
                    
                    move_idx = self.neural_predictor.move_to_int[move_uci]
                    neural_priority = probabilities[move_idx] * 1000  # Scale up probability
                    priority += neural_priority
            except Exception:
                pass  # Fall back to traditional ordering if neural network fails
            
            return priority
        
        # Sort moves by priority (highest first)
        moves.sort(key=move_priority, reverse=True)
        return moves
    
    def minimax(self, board: Board, depth: int, alpha: int, beta: int, 
                maximizing_player: bool) -> Tuple[int, Optional[Move]]:
        """
        Minimax algorithm with alpha-beta pruning and neural guidance
        
        Args:
            board: Current board position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: True if maximizing, False if minimizing
            
        Returns:
            tuple: (best_score, best_move)
        """
        self.nodes_searched += 1
        
        # Terminal node evaluation
        if depth == 0 or board.is_game_over():
            return self.hybrid_evaluation(board), None
        
        best_move = None
        
        # Use neural-guided move ordering if available, otherwise traditional
        if self.neural_predictor:
            ordered_moves = self.neural_move_ordering(board)
        else:
            ordered_moves = self.traditional_move_ordering(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval, best_move
    
    def traditional_move_ordering(self, board: Board) -> List[Move]:
        """Traditional move ordering (fallback when neural network unavailable)"""
        moves = list(board.legal_moves)
        
        def move_priority(move):
            priority = 0
            
            # Captures get high priority
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    priority += self.PIECE_VALUES[captured_piece.piece_type]
                
                # MVV-LVA
                attacker = board.piece_at(move.from_square)
                if attacker:
                    priority -= self.PIECE_VALUES[attacker.piece_type] // 10
            
            # Checks get medium priority
            board.push(move)
            if board.is_check():
                priority += 50
            board.pop()
            
            # Promotions get high priority
            if move.promotion:
                priority += 800
            
            return priority
        
        moves.sort(key=move_priority, reverse=True)
        return moves
    
    def get_best_move(self, board: Board, time_limit=None) -> Optional[str]:
        """
        Get best move using alpha-beta search with neural guidance
        
        Args:
            board: Current board position
            time_limit: Maximum time to search (seconds)
            
        Returns:
            str: UCI move string or None
        """
        if board.is_game_over():
            return None
        
        self.nodes_searched = 0
        start_time = time.time()
        
        # Iterative deepening
        best_move = None
        for depth in range(1, self.max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
            _, move = self.minimax(board, depth, float('-inf'), float('inf'), 
                                 board.turn)
            if move:
                best_move = move
        
        search_time = time.time() - start_time
        
        if best_move:
            engine_type = "Neural-Enhanced Alpha-Beta" if self.neural_predictor else "Alpha-Beta"
            print(f"{engine_type}: {best_move.uci()} (depth {min(depth, self.max_depth)}, "
                  f"{self.nodes_searched} nodes, {search_time:.3f}s)")
            return best_move.uci()
        
        # Fallback to first legal move
        legal_moves = list(board.legal_moves)
        return legal_moves[0].uci() if legal_moves else None


def demo():
    """Demo the alpha-beta engine"""
    print("Fixed Alpha-Beta Engine Demo")
    print("="*50)
    
    # Test without neural network
    print("\n1. Traditional Alpha-Beta:")
    traditional_engine = AlphaBetaEngine(max_depth=4, neural_weight=0.0)
    board = Board()
    move1 = traditional_engine.get_best_move(board, time_limit=2.0)
    print(f"Move: {move1}")
    
    # Test with neural network (if available)
    print("\n2. Neural-Enhanced Alpha-Beta:")
    try:
        enhanced_engine = AlphaBetaEngine(
            max_depth=4, 
            neural_weight=0.4  # 40% neural, 60% traditional
        )
        move2 = enhanced_engine.get_best_move(board, time_limit=2.0)
        print(f"Move: {move2}")
        
        if enhanced_engine.neural_predictor:
            if move1 != move2:
                print(f"✓ Neural guidance changed the move choice!")
            else:
                print(f"- Both engines chose the same move")
        else:
            print(f"✗ Neural predictor failed to load")
            
    except Exception as e:
        print(f"Neural enhancement not available: {e}")
    
    print("\nStarting position:")
    print(board)


if __name__ == "__main__":
    demo()