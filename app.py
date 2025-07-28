#!/usr/bin/env python3
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

# Add engines/torch directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
engines_path = os.path.join(current_dir, 'engines', 'torch')
if engines_path not in sys.path:
    sys.path.insert(0, engines_path)

# Import chess engines with better error handling
try:
    from predict import ChessPredictor
    from alphabeta_engine import AlphaBetaEngine
    print(f"✓ Successfully imported chess engines from: {engines_path}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Looking in: {engines_path}")
    print("Files in engines/torch:")
    try:
        files = os.listdir(engines_path)
        for f in files:
            print(f"  - {f}")
    except:
        print("  Directory not found or not accessible")
    print("\nMake sure you're running from the pilot-chess root directory")
    sys.exit(1)

# Stockfish integration (optional)
try:
    from chess.engine import SimpleEngine, Limit
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False
    print("ℹ️  Stockfish not available (optional)")

app = Flask(__name__)
CORS(app)

class ChessGameSession:
    """Individual chess game session"""
    
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
        """Make a move and update activity"""
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
        """Reset the game"""
        self.board = Board()
        self.game_history = []
        self.last_activity = datetime.now()
    
    def export_pgn(self) -> str:
        """Export game as PGN string"""
        game = pgn.Game()
        game.setup(Board())
        
        # Add metadata
        game.headers["Event"] = "Chess Engine Game"
        game.headers["Site"] = "Chess Engine API"
        game.headers["Date"] = self.created_at.strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Player"
        game.headers["Black"] = "Engine"
        
        # Add result
        if self.board.is_game_over():
            if self.board.is_checkmate():
                result = "1-0" if not self.board.turn else "0-1"
            else:
                result = "1/2-1/2"
        else:
            result = "*"
        game.headers["Result"] = result
        
        # Add moves
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


class ChessEngineManager:
    """Manages chess engines"""
    
    def __init__(self):
        self.engines = {}
        self.stockfish_path = None
        self.initialize_engines()
    
    def initialize_engines(self, neural_model_path=None, neural_mapping_path=None,
                          alphabeta_depth=4, stockfish_path=None, neural_weight=0.4):
        """Initialize available engines"""
        print("Initializing engines...")
        
        # Pure neural network engine
        try:
            if neural_model_path and neural_mapping_path:
                self.engines['neural'] = ChessPredictor(neural_model_path, neural_mapping_path)
            else:
                self.engines['neural'] = ChessPredictor()
            print("✓ Neural network engine loaded")
        except Exception as e:
            print(f"✗ Failed to load neural engine: {e}")
        
        # Traditional alpha-beta engine
        try:
            self.engines['traditional'] = AlphaBetaEngine(
                max_depth=alphabeta_depth,
                neural_weight=0.0
            )
            print("✓ Traditional alpha-beta engine loaded")
        except Exception as e:
            print(f"✗ Failed to load traditional alpha-beta engine: {e}")
        
        # Neural-enhanced alpha-beta engine
        try:
            self.engines['alphabeta'] = AlphaBetaEngine(
                max_depth=alphabeta_depth,
                neural_model_path=neural_model_path,
                neural_mapping_path=neural_mapping_path,
                neural_weight=neural_weight
            )
            print("✓ Neural-enhanced alpha-beta engine loaded")
        except Exception as e:
            print(f"✗ Failed to load neural-enhanced alpha-beta engine: {e}")
        
        # Stockfish engine
        if stockfish_path and os.path.exists(stockfish_path) and STOCKFISH_AVAILABLE:
            try:
                self.engines['stockfish'] = SimpleEngine.popen_uci(stockfish_path)
                self.stockfish_path = stockfish_path
                print("✓ Stockfish engine loaded")
            except Exception as e:
                print(f"✗ Failed to load Stockfish: {e}")
    
    def get_engine_move(self, engine_name: str, board: Board, time_limit=5.0) -> Optional[str]:
        """Get move from specified engine"""
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
        """Get list of available engines"""
        return list(self.engines.keys())
    
    def cleanup(self):
        """Clean up resources"""
        if 'stockfish' in self.engines:
            try:
                self.engines['stockfish'].quit()
            except:
                pass


# Global instances
engine_manager = ChessEngineManager()
game_sessions: Dict[str, ChessGameSession] = {}

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engines': engine_manager.get_available_engines(),
        'active_sessions': len(game_sessions)
    })

@app.route('/api/engines', methods=['GET'])
def get_engines():
    """Get available engines"""
    return jsonify({
        'engines': engine_manager.get_available_engines()
    })

@app.route('/api/game/new', methods=['POST'])
def create_game():
    """Create a new game session"""
    session_id = str(uuid.uuid4())
    game_sessions[session_id] = ChessGameSession(session_id)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'game_state': game_sessions[session_id].to_dict()
    })

@app.route('/api/game/<session_id>', methods=['GET'])
def get_game_state(session_id):
    """Get current game state"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    return jsonify({
        'success': True,
        'game_state': game_sessions[session_id].to_dict()
    })

@app.route('/api/game/<session_id>/move', methods=['POST'])
def make_move(session_id):
    """Make a move in the game"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    data = request.get_json()
    move_uci = data.get('move')
    
    if not move_uci:
        return jsonify({'error': 'Move is required'}), 400
    
    game = game_sessions[session_id]
    success = game.make_move(move_uci)
    
    if not success:
        return jsonify({'error': 'Invalid move'}), 400
    
    return jsonify({
        'success': True,
        'game_state': game.to_dict()
    })

@app.route('/api/game/<session_id>/engine-move', methods=['POST'])
def get_engine_move(session_id):
    """Get move from engine"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    data = request.get_json()
    engine_name = data.get('engine', 'alphabeta')
    time_limit = data.get('time_limit', 5.0)
    
    game = game_sessions[session_id]
    move = engine_manager.get_engine_move(engine_name, game.board, time_limit)
    
    if not move:
        return jsonify({'error': 'Engine could not find a move'}), 500
    
    # Make the engine move
    success = game.make_move(move)
    
    if not success:
        return jsonify({'error': 'Engine made invalid move'}), 500
    
    return jsonify({
        'success': True,
        'move': move,
        'game_state': game.to_dict()
    })

@app.route('/api/game/<session_id>/reset', methods=['POST'])
def reset_game(session_id):
    """Reset the game"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    game_sessions[session_id].reset()
    
    return jsonify({
        'success': True,
        'game_state': game_sessions[session_id].to_dict()
    })

@app.route('/api/game/<session_id>/pgn', methods=['GET'])
def export_pgn(session_id):
    """Export game as PGN"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    game = game_sessions[session_id]
    pgn_content = game.export_pgn()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pgn')
    temp_file.write(pgn_content)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f'chess_game_{session_id[:8]}.pgn',
        mimetype='application/x-chess-pgn'
    )

@app.route('/api/game/<session_id>/pgn-content', methods=['GET'])
def get_pgn_content(session_id):
    """Get PGN content as JSON"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    game = game_sessions[session_id]
    pgn_content = game.export_pgn()
    
    return jsonify({
        'success': True,
        'pgn': pgn_content
    })

@app.route('/api/game/<session_id>/compare-engines', methods=['POST'])
def compare_engines(session_id):
    """Compare all engines on current position"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    data = request.get_json()
    time_limit = data.get('time_limit', 3.0)
    
    game = game_sessions[session_id]
    results = {}
    
    import time
    for engine_name in engine_manager.get_available_engines():
        start_time = time.time()
        move = engine_manager.get_engine_move(engine_name, game.board, time_limit)
        elapsed = time.time() - start_time
        
        results[engine_name] = {
            'move': move,
            'time': elapsed
        }
    
    return jsonify({
        'success': True,
        'results': results,
        'position_fen': game.board.fen()
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, game in game_sessions.items():
        sessions.append({
            'session_id': session_id,
            'created_at': game.created_at.isoformat(),
            'last_activity': game.last_activity.isoformat(),
            'move_count': len(game.game_history),
            'is_game_over': game.board.is_game_over()
        })
    
    return jsonify({
        'success': True,
        'sessions': sessions
    })

@app.route('/api/game/<session_id>/delete', methods=['DELETE'])
def delete_session(session_id):
    """Delete a game session"""
    if session_id not in game_sessions:
        return jsonify({'error': 'Game session not found'}), 404
    
    del game_sessions[session_id]
    
    return jsonify({
        'success': True,
        'message': 'Session deleted'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# This allows the app to be imported and run from run_api.py
if __name__ == '__main__':
    print("Please use run_app.py to start the server from the pilot-chess root directory")
    print("Example: python run_app.py --help")