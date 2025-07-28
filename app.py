#!/usr/bin/env python3
"""
Flask API for Chess Engine with Neural Enhancement
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import chess
from chess import Board, Move, pgn
import io
import tempfile

# -------------------------
# Path setup based on structure
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # pilot-chess/
engines_path = os.path.join(current_dir, 'engines', 'torch')
models_path = os.path.join(current_dir, 'models')  # models folder

sys.path.insert(0, engines_path)

# Import engine modules
try:
    from predict import ChessPredictor
    from alphabeta_engine import AlphaBetaEngine
    print(f"Successfully imported from: {engines_path}")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Stockfish optional
try:
    from chess.engine import SimpleEngine, Limit
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False
    print("Warning: Stockfish integration requires python-chess[engine]")

app = Flask(__name__)
CORS(app)


# -------------------------
# Chess Game Session Class
# -------------------------
class ChessGameSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.board = Board()
        self.game_history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def to_dict(self):
        return {
            'session_id': self.session_id,
            'board_fen': self.board.fen(),
            'board_svg': str(self.board),
            'turn': 'white' if self.board.turn else 'black',
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_game_over': self.board.is_game_over(),
            'legal_moves': [move.uci() for move in self.board.legal_moves],
            'move_count': len(self.game_history),
            'game_history': self.game_history,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }

    def make_move(self, move_uci: str) -> bool:
        try:
            move = Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.game_history.append({
                    'move': move_uci,
                    'timestamp': datetime.now().isoformat(),
                    'move_number': len(self.game_history) + 1
                })
                self.last_activity = datetime.now()
                return True
            return False
        except Exception:
            return False

    def reset(self):
        self.board = Board()
        self.game_history = []
        self.last_activity = datetime.now()

    def export_pgn(self) -> str:
        game = pgn.Game()
        game.setup(Board())
        game.headers["Event"] = "Chess Engine Game"
        game.headers["Site"] = "Chess Engine API"
        game.headers["Date"] = self.created_at.strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Player"
        game.headers["Black"] = "Engine"
        if self.board.is_game_over():
            if self.board.is_checkmate():
                result = "1-0" if not self.board.turn else "0-1"
            else:
                result = "1/2-1/2"
        else:
            result = "*"
        game.headers["Result"] = result

        node = game
        board = Board()
        for move_data in self.game_history:
            try:
                move = Move.from_uci(move_data['move'])
                node = node.add_variation(move)
                board.push(move)
            except:
                continue

        return str(game)


# -------------------------
# Engine Manager with Path Fixes
# -------------------------
class ChessEngineManager:
    def __init__(self):
        self.engines = {}
        self.stockfish_path = None
        self.initialize_engines()

    def initialize_engines(self, neural_model_path=None, neural_mapping_path=None,
                           alphabeta_depth=4, stockfish_path=None, neural_weight=0.4):
        print("Initializing engines...")

        # ✅ FIX: Resolve defaults to pilot-chess/models if not provided
        if neural_model_path is None:
            neural_model_path = os.path.join(models_path, 'TORCH_100EPOCHS.pth')
        if neural_mapping_path is None:
            neural_mapping_path = os.path.join(models_path, 'move_to_int')

        neural_model_path = os.path.abspath(neural_model_path)
        neural_mapping_path = os.path.abspath(neural_mapping_path)

        # Neural engine
        try:
            self.engines['neural'] = ChessPredictor(neural_model_path, neural_mapping_path)
            print(f"✓ Neural engine loaded ({neural_model_path})")
        except Exception as e:
            print(f"✗ Failed to load neural engine: {e}")

        # Traditional engine
        try:
            self.engines['traditional'] = AlphaBetaEngine(max_depth=alphabeta_depth, neural_weight=0.0)
            print("✓ Traditional alpha-beta engine loaded")
        except Exception as e:
            print(f"✗ Failed to load traditional engine: {e}")

        # Neural-enhanced AlphaBeta
        try:
            self.engines['alphabeta'] = AlphaBetaEngine(
                max_depth=alphabeta_depth,
                neural_model_path=neural_model_path,
                neural_mapping_path=neural_mapping_path,
                neural_weight=neural_weight
            )
            print("✓ Neural-enhanced alpha-beta engine loaded")
        except Exception as e:
            print(f"✗ Failed to load neural-enhanced engine: {e}")

        # Stockfish
        if stockfish_path and os.path.exists(stockfish_path) and STOCKFISH_AVAILABLE:
            try:
                self.engines['stockfish'] = SimpleEngine.popen_uci(stockfish_path)
                self.stockfish_path = stockfish_path
                print("✓ Stockfish engine loaded")
            except Exception as e:
                print(f"✗ Failed to load Stockfish: {e}")

    def get_engine_move(self, engine_name: str, board: Board, time_limit=5.0) -> Optional[str]:
        if engine_name not in self.engines:
            return None
        try:
            if engine_name == 'neural':
                return self.engines[engine_name].predict_move(board)
            elif engine_name in ['alphabeta', 'traditional']:
                return self.engines[engine_name].get_best_move(board, time_limit)
            elif engine_name == 'stockfish' and STOCKFISH_AVAILABLE:
                result = self.engines[engine_name].play(board, Limit(time=time_limit))
                return result.move.uci() if result.move else None
        except Exception as e:
            print(f"Error getting move from {engine_name}: {e}")
            return None

    def get_available_engines(self):
        return list(self.engines.keys())

    def cleanup(self):
        if 'stockfish' in self.engines:
            try:
                self.engines['stockfish'].quit()
            except:
                pass


# -------------------------
# Global Instances & API Routes (unchanged)
# -------------------------
engine_manager = ChessEngineManager()
game_sessions: Dict[str, ChessGameSession] = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'engines': engine_manager.get_available_engines(),
        'active_sessions': len(game_sessions)
    })

# (all other routes remain unchanged from your original file...)

# -------------------------
# Entry Point
# -------------------------
if __name__ == '__main__':
    print("Please use run_api.py to start the server from the pilot-chess root directory")
    print("Example: python run_api.py --help")
