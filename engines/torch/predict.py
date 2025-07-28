#!/usr/bin/env python3
import torch
import pickle
import numpy as np
from chess import Board, Move
from auxiliary_func import board_to_matrix
from model import ChessModel


class ChessPredictor:
    """Chess move prediction using trained neural network"""
    
    def __init__(self, model_path="../../models/TORCH_100EPOCHS.pth", 
                 mapping_path="../../models/move_to_int"):
        """
        Initialize predictor with trained model and move mapping
        
        Args:
            model_path: Path to trained model weights
            mapping_path: Path to move encoding mapping
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        
        # Load move mapping
        with open(mapping_path, "rb") as file:
            self.move_to_int = pickle.load(file)
        
        self.int_to_move = {v: k for k, v in self.move_to_int.items()}
        self.num_classes = len(self.move_to_int)
        
        # Load and initialize model
        self.model = ChessModel(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with {self.num_classes} possible moves")
    
    def prepare_input(self, board: Board):
        """Convert board to model input tensor"""
        matrix = board_to_matrix(board)
        return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    
    def predict_move(self, board: Board, temperature=1.0):
        """
        Predict best legal move for given board position
        
        Args:
            board: Chess board position
            temperature: Temperature for probability scaling (1.0 = no scaling)
            
        Returns:
            str: UCI move string (e.g., 'e2e4') or None if no legal move found
        """
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        logits = logits.squeeze(0)  # Remove batch dimension
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        # Sort moves by probability (highest first)
        sorted_indices = np.argsort(probabilities)[::-1]
        
        for move_index in sorted_indices:
            if move_index in self.int_to_move:
                move = self.int_to_move[move_index]
                if move in legal_moves_uci:
                    return move
        
        # Fallback: return any legal move if model prediction fails
        if legal_moves:
            return legal_moves[0].uci()
        
        return None
    
    def get_top_moves(self, board: Board, top_k=5):
        """
        Get top k legal moves with their probabilities
        
        Args:
            board: Chess board position
            top_k: Number of top moves to return
            
        Returns:
            list: List of (move_uci, probability) tuples
        """
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        # Get legal move probabilities
        legal_move_probs = []
        for move_uci in legal_moves_uci:
            if move_uci in self.move_to_int:
                move_idx = self.move_to_int[move_uci]
                prob = probabilities[move_idx]
                legal_move_probs.append((move_uci, prob))
        
        # Sort by probability and return top k
        legal_move_probs.sort(key=lambda x: x[1], reverse=True)
        return legal_move_probs[:top_k]


def demo():
    """Demo function to test prediction"""
    try:
        predictor = ChessPredictor()
        
        # Test with starting position
        board = Board()
        print("Starting position:")
        print(board)
        print()
        
        # Get top 3 moves
        top_moves = predictor.get_top_moves(board, top_k=3)
        print("Top 3 predicted moves:")
        for i, (move, prob) in enumerate(top_moves, 1):
            print(f"{i}. {move} (probability: {prob:.4f})")
        
        # Make the best move
        best_move = predictor.predict_move(board)
        if best_move:
            board.push_uci(best_move)
            print(f"\nAfter playing {best_move}:")
            print(board)
        else:
            print("No move predicted!")
            
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    demo()