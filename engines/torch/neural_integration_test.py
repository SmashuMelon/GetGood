#!/usr/bin/env python3
"""
Test script to verify that the alpha-beta engine is actually using neural network guidance
Compares pure traditional vs neural-enhanced alpha-beta behavior
"""

import sys
import os
import time
from chess import Board

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import ChessPredictor
    from alphabeta_engine import AlphaBetaEngine
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class NeuralIntegrationTest:
    """Test suite to verify neural integration in alpha-beta engine"""
    
    def __init__(self):
        self.engines = {}
        self.test_positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # After 1.e4
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            # Sicilian Defense position
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            # Middle game position
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        ]
    
    def initialize_engines(self):
        """Initialize engines for comparison"""
        print("=" * 70)
        print("NEURAL INTEGRATION VERIFICATION TEST")
        print("=" * 70)
        print("\n1. Initializing Engines for Comparison...")
        print("-" * 50)
        
        # Pure neural network (for reference)
        try:
            self.engines['pure_neural'] = ChessPredictor()
            print("✓ Pure neural network loaded")
        except Exception as e:
            print(f"✗ Pure neural network failed: {e}")
            return False
        
        # Traditional alpha-beta (NO neural guidance)
        try:
            self.engines['traditional'] = AlphaBetaEngine(
                max_depth=3, 
                neural_weight=0.0  # Pure traditional
            )
            print("✓ Traditional alpha-beta loaded (neural_weight=0.0)")
        except Exception as e:
            print(f"✗ Traditional alpha-beta failed: {e}")
            return False
        
        # Neural-enhanced alpha-beta (WITH neural guidance)
        try:
            self.engines['neural_enhanced'] = AlphaBetaEngine(
                max_depth=3, 
                neural_weight=0.5  # 50% neural guidance
            )
            print("✓ Neural-enhanced alpha-beta loaded (neural_weight=0.5)")
        except Exception as e:
            print(f"✗ Neural-enhanced alpha-beta failed: {e}")
            return False
        
        return True
    
    def check_neural_predictor_availability(self):
        """Check if the alpha-beta engines have neural predictors loaded"""
        print(f"\n2. Checking Neural Predictor Availability...")
        print("-" * 50)
        
        for name, engine in self.engines.items():
            if hasattr(engine, 'neural_predictor'):
                if engine.neural_predictor is not None:
                    print(f"✓ {name:20}: Neural predictor AVAILABLE")
                    print(f"  └─ Neural weight: {getattr(engine, 'neural_weight', 'N/A')}")
                    print(f"  └─ Device: {engine.neural_predictor.device}")
                    print(f"  └─ Classes: {engine.neural_predictor.num_classes}")
                else:
                    print(f"✗ {name:20}: Neural predictor is None")
                    print(f"  └─ Neural weight: {getattr(engine, 'neural_weight', 'N/A')}")
            else:
                print(f"- {name:20}: No neural predictor attribute (pure neural)")
    
    def test_move_differences(self):
        """Compare moves between traditional and neural-enhanced engines"""
        print(f"\n3. Comparing Move Choices...")
        print("-" * 50)
        
        differences_found = 0
        total_positions = 0
        
        for i, fen in enumerate(self.test_positions, 1):
            board = Board(fen)
            print(f"\nPosition {i}: {fen}")
            print("Board:")
            print(board)
            print()
            
            moves = {}
            times = {}
            
            # Get moves from each alpha-beta engine
            for engine_name in ['traditional', 'neural_enhanced']:
                if engine_name in self.engines:
                    print(f"  {engine_name:18}: ", end="", flush=True)
                    start_time = time.time()
                    
                    try:
                        move = self.engines[engine_name].get_best_move(board, time_limit=3.0)
                        elapsed = time.time() - start_time
                        moves[engine_name] = move
                        times[engine_name] = elapsed
                        print(f"{move} ({elapsed:.3f}s)")
                    except Exception as e:
                        moves[engine_name] = None
                        times[engine_name] = 0
                        print(f"ERROR: {e}")
            
            # Also get pure neural move for reference
            if 'pure_neural' in self.engines:
                print(f"  {'pure_neural':18}: ", end="", flush=True)
                try:
                    start_time = time.time()
                    move = self.engines['pure_neural'].predict_move(board)
                    elapsed = time.time() - start_time
                    moves['pure_neural'] = move
                    times['pure_neural'] = elapsed
                    print(f"{move} ({elapsed:.3f}s)")
                except Exception as e:
                    moves['pure_neural'] = None
                    times['pure_neural'] = 0
                    print(f"ERROR: {e}")
            
            total_positions += 1
            
            # Compare the moves
            trad_move = moves.get('traditional')
            neural_move = moves.get('neural_enhanced')
            pure_neural_move = moves.get('pure_neural')
            
            print(f"\n  Analysis:")
            if trad_move and neural_move:
                if trad_move != neural_move:
                    differences_found += 1
                    print(f"  ✓ DIFFERENCE FOUND: Traditional≠Neural-Enhanced")
                    print(f"    Traditional:      {trad_move}")
                    print(f"    Neural-Enhanced:  {neural_move}")
                    
                    if pure_neural_move:
                        if neural_move == pure_neural_move:
                            print(f"    Pure Neural:      {pure_neural_move} (matches neural-enhanced)")
                        else:
                            print(f"    Pure Neural:      {pure_neural_move} (different from both)")
                else:
                    print(f"  - Same move chosen: {trad_move}")
                    if pure_neural_move and pure_neural_move != trad_move:
                        print(f"    But pure neural suggests: {pure_neural_move}")
            else:
                print(f"  ✗ Could not compare (missing moves)")
            
            print("-" * 40)
        
        print(f"\nMove Comparison Summary:")
        print(f"  Total positions tested: {total_positions}")
        print(f"  Positions with different moves: {differences_found}")
        print(f"  Difference rate: {differences_found/total_positions*100:.1f}%")
        
        if differences_found > 0:
            print(f"  ✓ NEURAL INTEGRATION DETECTED: Engines make different choices")
        else:
            print(f"  ⚠ NO DIFFERENCES: Neural integration may not be working")
    
    def test_evaluation_differences(self):
        """Test if evaluation functions return different scores"""
        print(f"\n4. Testing Evaluation Function Differences...")
        print("-" * 50)
        
        board = Board()  # Starting position
        print(f"Testing evaluations on starting position...")
        
        try:
            # Test traditional evaluation
            if 'traditional' in self.engines:
                trad_eval = self.engines['traditional'].traditional_evaluation(board)
                print(f"  Traditional evaluation:      {trad_eval}")
            
            # Test neural-enhanced evaluation  
            if 'neural_enhanced' in self.engines:
                neural_eval = self.engines['neural_enhanced'].neural_evaluation(board)
                hybrid_eval = self.engines['neural_enhanced'].hybrid_evaluation(board)
                print(f"  Neural evaluation:           {neural_eval}")
                print(f"  Hybrid evaluation:           {hybrid_eval}")
                
                if abs(trad_eval - hybrid_eval) > 10:  # Significant difference
                    print(f"  ✓ SIGNIFICANT DIFFERENCE: Neural guidance is affecting evaluation")
                else:
                    print(f"  - Small difference: Neural guidance may be minimal")
        
        except Exception as e:
            print(f"  ✗ Error testing evaluations: {e}")
    
    def test_move_ordering(self):
        """Test if neural move ordering is working"""
        print(f"\n5. Testing Neural Move Ordering...")
        print("-" * 50)
        
        board = Board()
        
        try:
            if 'neural_enhanced' in self.engines:
                engine = self.engines['neural_enhanced']
                
                # Get traditional move ordering
                traditional_moves = engine.traditional_move_ordering(board)
                print(f"  Traditional move ordering (first 5):")
                for i, move in enumerate(traditional_moves[:5], 1):
                    print(f"    {i}. {move.uci()}")
                
                # Get neural move ordering (if available)
                if hasattr(engine, 'neural_move_ordering') and engine.neural_predictor:
                    neural_moves = engine.neural_move_ordering(board)
                    print(f"\n  Neural move ordering (first 5):")
                    for i, move in enumerate(neural_moves[:5], 1):
                        print(f"    {i}. {move.uci()}")
                    
                    # Compare orders
                    trad_first_5 = [m.uci() for m in traditional_moves[:5]]
                    neural_first_5 = [m.uci() for m in neural_moves[:5]]
                    
                    if trad_first_5 != neural_first_5:
                        print(f"  ✓ DIFFERENT MOVE ORDERING: Neural guidance is working")
                    else:
                        print(f"  - Same move ordering: Neural may not be significantly affecting order")
                else:
                    print(f"  - Neural move ordering not available or neural predictor missing")
        
        except Exception as e:
            print(f"  ✗ Error testing move ordering: {e}")
    
    def run_comprehensive_test(self):
        """Run all neural integration tests"""
        if not self.initialize_engines():
            print("Failed to initialize engines. Exiting.")
            return False
        
        self.check_neural_predictor_availability()
        self.test_move_differences()
        self.test_evaluation_differences()
        self.test_move_ordering()
        
        return True
    
    def generate_final_verdict(self):
        """Generate final verdict on neural integration"""
        print(f"\n" + "=" * 70)
        print("FINAL VERDICT: NEURAL INTEGRATION STATUS")
        print("=" * 70)
        
        # Check if neural predictors are loaded
        neural_enhanced_engine = self.engines.get('neural_enhanced')
        traditional_engine = self.engines.get('traditional')
        
        if neural_enhanced_engine and hasattr(neural_enhanced_engine, 'neural_predictor'):
            if neural_enhanced_engine.neural_predictor is not None:
                print("✓ Neural predictor is loaded in neural-enhanced engine")
                print(f"✓ Neural weight is set to: {neural_enhanced_engine.neural_weight}")
                
                if traditional_engine and hasattr(traditional_engine, 'neural_predictor'):
                    if traditional_engine.neural_predictor is None:
                        print("✓ Traditional engine correctly has NO neural predictor")
                    else:
                        print("⚠ Traditional engine unexpectedly has neural predictor")
                
                print("\n🎯 CONCLUSION: Neural integration appears to be WORKING")
                print("   The alpha-beta engine should be using neural network guidance")
                
                return True
            else:
                print("✗ Neural predictor is None in neural-enhanced engine")
                print("\n❌ CONCLUSION: Neural integration is NOT working")
                return False
        else:
            print("✗ Neural predictor attribute missing")
            print("\n❌ CONCLUSION: Neural integration is NOT working")
            return False


def main():
    """Run the comprehensive neural integration test"""
    test_suite = NeuralIntegrationTest()
    
    try:
        success = test_suite.run_comprehensive_test()
        test_suite.generate_final_verdict()
        
        if success:
            print(f"\n💡 TIP: If you see move differences between traditional and neural-enhanced")
            print(f"   engines, that's a good sign that neural integration is working!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()