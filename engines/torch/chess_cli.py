#!/usr/bin/env python3
import sys
import os
import argparse
import time
from typing import Optional, Dict, Any
import chess
from chess import Board, Move, pgn
try:
    from chess.engine import SimpleEngine, Limit
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False
    print("Warning: Stockfish integration requires python-chess with engine support")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import ChessPredictor
    from alphabeta_engine import AlphaBetaEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all engine files are in the same directory")
    sys.exit(1)


class ChessGame:
    """Main chess game controller with enhanced engines"""
    
    def __init__(self):
        self.board = Board()
        self.game_history = []
        self.engines = {}
        self.stockfish_path = None
        
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
        
        # Traditional alpha-beta engine (no neural guidance)
        try:
            self.engines['traditional'] = AlphaBetaEngine(
                max_depth=alphabeta_depth,
                neural_weight=0.0  # Pure traditional
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
        elif stockfish_path and not STOCKFISH_AVAILABLE:
            print("✗ Stockfish integration not available with current chess library version")
            print("  Install python-chess[engine] for Stockfish support")
        elif stockfish_path:
            print(f"✗ Stockfish not found at: {stockfish_path}")
    
    def get_engine_move(self, engine_name: str, time_limit=5.0) -> Optional[str]:
        """Get move from specified engine"""
        if engine_name not in self.engines:
            print(f"Engine '{engine_name}' not available")
            return None
        
        try:
            if engine_name == 'neural':
                return self.engines[engine_name].predict_move(self.board)
            elif engine_name in ['alphabeta', 'traditional']:
                return self.engines[engine_name].get_best_move(self.board, time_limit)
            elif engine_name == 'stockfish' and STOCKFISH_AVAILABLE:
                result = self.engines[engine_name].play(self.board, Limit(time=time_limit))
                return result.move.uci() if result.move else None
        except Exception as e:
            print(f"Error getting move from {engine_name}: {e}")
            return None
    
    def get_human_move(self) -> Optional[str]:
        """Get move from human player"""
        legal_moves = list(self.board.legal_moves)
        legal_uci = [move.uci() for move in legal_moves]
        
        while True:
            print(f"\nLegal moves: {', '.join(legal_uci[:10])}" + 
                  ("..." if len(legal_uci) > 10 else ""))
            
            move_input = input("Enter your move (UCI format, e.g., 'e2e4') or 'quit': ").strip()
            
            if move_input.lower() in ['quit', 'q', 'exit']:
                return None
            
            if move_input in legal_uci:
                return move_input
            
            print(f"Invalid move: {move_input}")
    
    def make_move(self, move_uci: str) -> bool:
        """Make a move on the board"""
        try:
            move = Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.game_history.append(move_uci)
                return True
            else:
                print(f"Illegal move: {move_uci}")
                return False
        except Exception as e:
            print(f"Error making move {move_uci}: {e}")
            return False
    
    def display_board(self, flip=False):
        """Display the current board"""
        print("\n" + "="*40)
        if flip:
            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal))
        else:
            print(self.board)
        print("="*40)
        
        # Show game info
        turn = "White" if self.board.turn else "Black"
        print(f"Turn: {turn}")
        
        if self.board.is_check():
            print("CHECK!")
        
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                print(f"CHECKMATE! {winner} wins!")
            elif self.board.is_stalemate():
                print("STALEMATE! Draw!")
            else:
                print("Game over! Draw!")
    
    def play_human_vs_engine(self, engine_name: str, human_is_white: bool = True):
        """Play human vs engine"""
        print(f"\n=== Human vs {engine_name.title()} Engine ===")
        print(f"You are playing as {'White' if human_is_white else 'Black'}")
        
        while not self.board.is_game_over():
            self.display_board(flip=not human_is_white)
            
            if self.board.turn == human_is_white:
                # Human's turn
                print(f"\nYour turn!")
                move = self.get_human_move()
                if move is None:
                    print("Game aborted by player")
                    break
                self.make_move(move)
            else:
                # Engine's turn
                print(f"\n{engine_name.title()} is thinking...")
                move = self.get_engine_move(engine_name)
                if move:
                    print(f"{engine_name.title()} plays: {move}")
                    self.make_move(move)
                else:
                    print(f"{engine_name.title()} couldn't find a move!")
                    break
        
        self.display_board(flip=not human_is_white)
        self.show_game_result()
    
    def play_engine_vs_engine(self, white_engine: str, black_engine: str, 
                             max_moves: int = 100, time_per_move: float = 2.0):
        """Play engine vs engine"""
        print(f"\n=== {white_engine.title()} (White) vs {black_engine.title()} (Black) ===")
        
        move_count = 0
        while not self.board.is_game_over() and move_count < max_moves:
            self.display_board()
            
            current_engine = white_engine if self.board.turn else black_engine
            player_name = f"{current_engine.title()} ({'White' if self.board.turn else 'Black'})"
            
            print(f"\n{player_name} is thinking...")
            move = self.get_engine_move(current_engine, time_per_move)
            
            if move:
                print(f"{player_name} plays: {move}")
                self.make_move(move)
                move_count += 1
                time.sleep(0.5)  # Brief pause for readability
            else:
                print(f"{player_name} couldn't find a move!")
                break
        
        if move_count >= max_moves:
            print(f"\nGame ended after {max_moves} moves")
        
        self.display_board()
        self.show_game_result()
    
    def compare_engines(self, position_fen=None, time_limit=3.0):
        """Compare all available engines on a position"""
        if position_fen:
            self.board = Board(position_fen)
        
        print(f"\n=== Engine Comparison ===")
        print(f"Position FEN: {self.board.fen()}")
        self.display_board()
        
        results = {}
        
        for engine_name in self.engines.keys():
            print(f"\n{engine_name.title()} thinking...")
            start_time = time.time()
            move = self.get_engine_move(engine_name, time_limit)
            elapsed = time.time() - start_time
            
            results[engine_name] = {
                'move': move,
                'time': elapsed
            }
        
        print(f"\n=== Results ===")
        for engine, result in results.items():
            print(f"{engine:15}: {result['move']:8} ({result['time']:.3f}s)")
    
    def show_game_result(self):
        """Display game result and statistics"""
        print("\n" + "="*40)
        print("GAME OVER")
        
        if self.board.is_checkmate():
            winner = "White" if not self.board.turn else "Black"
            print(f"Result: {winner} wins by checkmate!")
        elif self.board.is_stalemate():
            print("Result: Draw by stalemate")
        elif self.board.is_insufficient_material():
            print("Result: Draw by insufficient material")
        elif self.board.is_fifty_moves():
            print("Result: Draw by fifty-move rule")
        elif self.board.is_repetition():
            print("Result: Draw by repetition")
        else:
            print("Result: Game incomplete")
        
        print(f"Total moves: {len(self.game_history)}")
        print("="*40)
    
    def save_game(self, filename: str):
        """Save game to PGN file"""
        try:
            game = pgn.Game()
            game.setup(Board())
            
            node = game
            board = Board()
            for move_uci in self.game_history:
                move = Move.from_uci(move_uci)
                node = node.add_variation(move)
                board.push(move)
            
            with open(filename, 'w') as f:
                print(game, file=f)
            
            print(f"Game saved to {filename}")
        except Exception as e:
            print(f"Error saving game: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if 'stockfish' in self.engines:
            try:
                self.engines['stockfish'].quit()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Chess Engine CLI with Neural Enhancement')
    parser.add_argument('--mode', choices=['human-engine', 'engine-engine', 'demo', 'compare'],
                       default='human-engine', help='Game mode')
    parser.add_argument('--human-color', choices=['white', 'black'], default='white',
                       help='Human player color (for human-engine mode)')
    parser.add_argument('--white-engine', choices=['neural', 'alphabeta', 'traditional', 'stockfish'],
                       default='alphabeta', help='White player engine')
    parser.add_argument('--black-engine', choices=['neural', 'alphabeta', 'traditional', 'stockfish'],
                       default='traditional', help='Black player engine')
    parser.add_argument('--engine', choices=['neural', 'alphabeta', 'traditional', 'stockfish'],
                       default='alphabeta', help='Engine for human-engine mode')
    parser.add_argument('--neural-model', help='Path to neural network model')
    parser.add_argument('--neural-mapping', help='Path to neural network move mapping')
    parser.add_argument('--alphabeta-depth', type=int, default=4,
                       help='Alpha-beta search depth')
    parser.add_argument('--neural-weight', type=float, default=0.4,
                       help='Weight for neural evaluation (0.0-1.0, default 0.4)')
    parser.add_argument('--stockfish-path', help='Path to Stockfish executable')
    parser.add_argument('--time-per-move', type=float, default=2.0,
                       help='Time limit per move (seconds)')
    parser.add_argument('--max-moves', type=int, default=100,
                       help='Maximum moves for engine-engine games')
    parser.add_argument('--save-game', help='Save game to PGN file')
    parser.add_argument('--position', help='Starting position FEN (for compare mode)')
    
    args = parser.parse_args()
    
    # Create game instance
    game = ChessGame()
    
    try:
        # Initialize engines
        game.initialize_engines(
            neural_model_path=args.neural_model,
            neural_mapping_path=args.neural_mapping,
            alphabeta_depth=args.alphabeta_depth,
            stockfish_path=args.stockfish_path,
            neural_weight=args.neural_weight
        )
        
        available_engines = list(game.engines.keys())
        print(f"\nAvailable engines: {available_engines}")
        
        # Run selected mode
        if args.mode == 'human-engine':
            if args.engine not in game.engines:
                print(f"Error: Engine '{args.engine}' not available")
                print(f"Available engines: {available_engines}")
                sys.exit(1)
            
            human_is_white = (args.human_color == 'white')
            game.play_human_vs_engine(args.engine, human_is_white)
            
        elif args.mode == 'engine-engine':
            if args.white_engine not in game.engines:
                print(f"Error: White engine '{args.white_engine}' not available")
                print(f"Available engines: {available_engines}")
                sys.exit(1)
            if args.black_engine not in game.engines:
                print(f"Error: Black engine '{args.black_engine}' not available")
                print(f"Available engines: {available_engines}")
                sys.exit(1)
            
            game.play_engine_vs_engine(args.white_engine, args.black_engine,
                                     args.max_moves, args.time_per_move)
            
        elif args.mode == 'compare':
            if len(game.engines) < 2:
                print("Need at least 2 engines for comparison")
                sys.exit(1)
            game.compare_engines(args.position, args.time_per_move)
            
        elif args.mode == 'demo':
            print("\n=== Demo Mode ===")
            print("Demonstrating different engines...")
            
            # Show starting position
            print("\nStarting position:")
            game.display_board()
            
            # Demo each available engine
            for engine_name in available_engines:
                print(f"\n--- {engine_name.title()} Engine ---")
                move = game.get_engine_move(engine_name, 3.0)
                if move:
                    print(f"Suggested move: {move}")
                    
                    # Show position after this move
                    test_board = game.board.copy()
                    test_board.push_uci(move)
                    print("Position after move:")
                    print(test_board)
                else:
                    print("No move found")
                print("-" * 30)
            
            # Compare all engines
            if len(available_engines) > 1:
                print("\n--- Engine Comparison ---")
                game.compare_engines(time_limit=3.0)
        
        # Save game if requested
        if args.save_game and game.game_history:
            game.save_game(args.save_game)
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        game.cleanup()


if __name__ == "__main__":
    main()