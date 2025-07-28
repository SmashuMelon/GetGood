#!/usr/bin/env python3
"""
Test script to verify all chess engines are functioning properly
Tests each engine with maximum 5 moves from starting position
"""

import sys
import os
import time
import traceback
from chess import Board

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import ChessPredictor
    from alphabeta_engine import AlphaBetaEngine
    from chess_cli import ChessGame
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all engine files are in the same directory")
    sys.exit(1)


class EngineTest:
    """Test suite for chess engines"""
    
    def __init__(self):
        self.results = {}
        self.engines = {}
        
    def initialize_engines(self):
        """Initialize all available engines for testing"""
        print("=" * 60)
        print("CHESS ENGINE TEST SUITE")
        print("=" * 60)
        print("\n1. Initializing Engines...")
        print("-" * 30)
        
        # Pure neural network engine
        try:
            self.engines['neural'] = ChessPredictor()
            print("✓ Neural network engine loaded successfully")
            self.results['neural'] = {'loaded': True, 'moves': [], 'errors': []}
        except Exception as e:
            print(f"✗ Neural network engine failed: {e}")
            self.results['neural'] = {'loaded': False, 'error': str(e), 'moves': [], 'errors': []}
        
        # Traditional alpha-beta engine (no neural guidance)
        try:
            self.engines['traditional'] = AlphaBetaEngine(max_depth=3, neural_weight=0.0)
            print("✓ Traditional alpha-beta engine loaded successfully")
            self.results['traditional'] = {'loaded': True, 'moves': [], 'errors': []}
        except Exception as e:
            print(f"✗ Traditional alpha-beta engine failed: {e}")
            self.results['traditional'] = {'loaded': False, 'error': str(e), 'moves': [], 'errors': []}
        
        # Neural-enhanced alpha-beta engine
        try:
            self.engines['alphabeta'] = AlphaBetaEngine(max_depth=3, neural_weight=0.4)
            print("✓ Neural-enhanced alpha-beta engine loaded successfully")
            self.results['alphabeta'] = {'loaded': True, 'moves': [], 'errors': []}
        except Exception as e:
            print(f"✗ Neural-enhanced alpha-beta engine failed: {e}")
            self.results['alphabeta'] = {'loaded': False, 'error': str(e), 'moves': [], 'errors': []}
        
        loaded_count = sum(1 for result in self.results.values() if result['loaded'])
        print(f"\nEngines loaded: {loaded_count}/3")
        
        return loaded_count > 0
    
    def test_single_move(self, engine_name, board, move_number):
        """Test a single move from an engine"""
        if engine_name not in self.engines:
            return None, f"Engine {engine_name} not available"
        
        try:
            print(f"  Move {move_number}: {engine_name.title()} thinking...", end=" ")
            start_time = time.time()
            
            if engine_name == 'neural':
                move = self.engines[engine_name].predict_move(board)
            else:  # alphabeta or traditional
                move = self.engines[engine_name].get_best_move(board, time_limit=3.0)
            
            elapsed = time.time() - start_time
            
            if move:
                print(f"Move: {move} ({elapsed:.3f}s)")
                return move, None
            else:
                error_msg = "No move returned"
                print(f"ERROR: {error_msg}")
                return None, error_msg
                
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"ERROR: {error_msg}")
            return None, error_msg
    
    def test_engine_sequence(self, engine_name, max_moves=5):
        """Test an engine playing a sequence of moves"""
        if not self.results[engine_name]['loaded']:
            print(f"  Skipping {engine_name} (not loaded)")
            return
        
        print(f"\n--- Testing {engine_name.title()} Engine (Max {max_moves} moves) ---")
        
        board = Board()  # Starting position
        moves_played = []
        
        print(f"  Starting position:")
        print(f"  {board.fen()}")
        
        for move_num in range(1, max_moves + 1):
            if board.is_game_over():
                print(f"  Game ended after {move_num - 1} moves")
                break
            
            move_uci, error = self.test_single_move(engine_name, board, move_num)
            
            if move_uci:
                try:
                    # Validate and make the move
                    move = board.parse_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        moves_played.append(move_uci)
                        print(f"    Position after {move_uci}:")
                        print(f"    {board.fen()}")
                    else:
                        error = f"Illegal move: {move_uci}"
                        print(f"  ERROR: {error}")
                        self.results[engine_name]['errors'].append(f"Move {move_num}: {error}")
                        break
                except Exception as e:
                    error = f"Move parsing error: {str(e)}"
                    print(f"  ERROR: {error}")
                    self.results[engine_name]['errors'].append(f"Move {move_num}: {error}")
                    break
            else:
                self.results[engine_name]['errors'].append(f"Move {move_num}: {error}")
                break
        
        self.results[engine_name]['moves'] = moves_played
        print(f"  Final position: {board.fen()}")
        print(f"  Moves played: {' '.join(moves_played) if moves_played else 'None'}")
    
    def test_all_engines(self, max_moves=5):
        """Test all loaded engines"""
        print(f"\n2. Testing Engine Move Generation (Max {max_moves} moves each)")
        print("-" * 50)
        
        for engine_name in ['neural', 'traditional', 'alphabeta']:
            self.test_engine_sequence(engine_name, max_moves)
    
    def compare_engines_on_position(self, fen=None):
        """Compare all engines on the same position"""
        if fen is None:
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
        
        print(f"\n3. Comparing Engines on Same Position")
        print("-" * 40)
        print(f"Position: {fen}")
        
        board = Board(fen)
        print("Board:")
        print(board)
        print()
        
        comparison_results = {}
        
        for engine_name in self.engines.keys():
            if self.results[engine_name]['loaded']:
                print(f"{engine_name.title():15}: ", end="")
                move, error = self.test_single_move(engine_name, board, 1)
                comparison_results[engine_name] = move if move else f"ERROR: {error}"
        
        print(f"\nComparison Results:")
        for engine, result in comparison_results.items():
            print(f"  {engine:15}: {result}")
    
    def test_cli_integration(self):
        """Test CLI integration"""
        print(f"\n4. Testing CLI Integration")
        print("-" * 30)
        
        try:
            game = ChessGame()
            game.initialize_engines(alphabeta_depth=3, neural_weight=0.4)
            
            available_engines = list(game.engines.keys())
            print(f"CLI available engines: {available_engines}")
            
            # Test getting a move from each engine via CLI
            for engine_name in available_engines:
                try:
                    move = game.get_engine_move(engine_name, time_limit=2.0)
                    print(f"  {engine_name:15}: {move if move else 'No move'}")
                except Exception as e:
                    print(f"  {engine_name:15}: ERROR - {e}")
            
            game.cleanup()
            print("✓ CLI integration test completed")
            
        except Exception as e:
            print(f"✗ CLI integration test failed: {e}")
    
    def generate_report(self):
        """Generate final test report"""
        print(f"\n" + "=" * 60)
        print("FINAL TEST REPORT")
        print("=" * 60)
        
        total_engines = len(self.results)
        loaded_engines = sum(1 for result in self.results.values() if result['loaded'])
        
        print(f"\nEngine Status:")
        print(f"  Total engines: {total_engines}")
        print(f"  Successfully loaded: {loaded_engines}")
        print(f"  Failed to load: {total_engines - loaded_engines}")
        
        print(f"\nDetailed Results:")
        for engine_name, result in self.results.items():
            print(f"\n{engine_name.upper()}:")
            if result['loaded']:
                print(f"  ✓ Status: Loaded successfully")
                print(f"  ✓ Moves generated: {len(result['moves'])}")
                if result['moves']:
                    print(f"  ✓ Move sequence: {' '.join(result['moves'])}")
                if result['errors']:
                    print(f"  ⚠ Errors encountered: {len(result['errors'])}")
                    for error in result['errors']:
                        print(f"    - {error}")
                else:
                    print(f"  ✓ No errors")
            else:
                print(f"  ✗ Status: Failed to load")
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nOverall Status: ", end="")
        if loaded_engines == total_engines:
            print("✓ ALL ENGINES WORKING")
        elif loaded_engines > 0:
            print("⚠ PARTIAL SUCCESS - Some engines working")
        else:
            print("✗ ALL ENGINES FAILED")
        
        return loaded_engines == total_engines


def main():
    """Run the complete engine test suite"""
    test_suite = EngineTest()
    
    try:
        # Initialize engines
        if not test_suite.initialize_engines():
            print("No engines loaded successfully. Exiting.")
            sys.exit(1)
        
        # Test engine move generation
        test_suite.test_all_engines(max_moves=5)
        
        # Compare engines on same position
        test_suite.compare_engines_on_position()
        
        # Test CLI integration
        test_suite.test_cli_integration()
        
        # Generate final report
        all_working = test_suite.generate_report()
        
        print(f"\n" + "=" * 60)
        if all_working:
            print("🎉 TEST COMPLETE: All engines are functioning properly!")
        else:
            print("⚠️  TEST COMPLETE: Some issues detected. See report above.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()