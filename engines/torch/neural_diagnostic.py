#!/usr/bin/env python3
"""
Quick test to verify neural integration is working
"""

import sys
import os
from chess import Board

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_neural_integration():
    """Quick test of neural integration"""
    print("=" * 60)
    print("QUICK NEURAL INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Import the fixed engine
        from alphabeta_engine import AlphaBetaEngine
        
        print("\n1. Creating engines...")
        
        # Create traditional engine (no neural)
        print("   Creating traditional engine...")
        traditional = AlphaBetaEngine(max_depth=3, neural_weight=0.0)
        print(f"   └─ Neural predictor: {traditional.neural_predictor}")
        
        # Create neural-enhanced engine
        print("   Creating neural-enhanced engine...")
        enhanced = AlphaBetaEngine(max_depth=3, neural_weight=0.5)
        print(f"   └─ Neural predictor: {enhanced.neural_predictor}")
        print(f"   └─ Neural weight: {enhanced.neural_weight}")
        
        if enhanced.neural_predictor is None:
            print("\n❌ FAILED: Neural predictor is None")
            print("   This means neural integration is NOT working")
            return False
        
        print("\n✅ SUCCESS: Neural predictor loaded successfully!")
        
        print("\n2. Testing move predictions...")
        board = Board()  # Starting position
        
        # Get moves from both engines
        print("   Traditional engine thinking...")
        trad_move = traditional.get_best_move(board, time_limit=2.0)
        
        print("   Neural-enhanced engine thinking...")
        neural_move = enhanced.get_best_move(board, time_limit=2.0)
        
        print(f"\n   Results:")
        print(f"   Traditional:      {trad_move}")
        print(f"   Neural-Enhanced:  {neural_move}")
        
        if trad_move != neural_move:
            print(f"\n✅ EXCELLENT: Different moves chosen!")
            print(f"   This confirms neural integration is working")
        else:
            print(f"\n⚠️  SAME MOVES: Both engines chose the same move")
            print(f"   This doesn't necessarily mean neural integration failed")
            print(f"   (they might just agree on the best move)")
        
        print("\n3. Testing neural evaluation...")
        trad_eval = traditional.traditional_evaluation(board)
        neural_eval = enhanced.neural_evaluation(board)
        hybrid_eval = enhanced.hybrid_evaluation(board)
        
        print(f"   Traditional eval: {trad_eval}")
        print(f"   Neural eval:      {neural_eval}")
        print(f"   Hybrid eval:      {hybrid_eval}")
        
        if abs(hybrid_eval - trad_eval) > 10:
            print(f"\n✅ GREAT: Significant difference in evaluation!")
            print(f"   Neural guidance is affecting the evaluation")
        else:
            print(f"\n- Small evaluation difference")
        
        print("\n" + "=" * 60)
        print("FINAL RESULT: Neural integration appears to be WORKING! ✅")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Make sure all files are in the same directory")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the quick test"""
    success = test_neural_integration()
    
    if success:
        print("\n🎉 Neural integration test PASSED!")
        print("   You can now run the full neural_integration_test.py")
    else:
        print("\n💔 Neural integration test FAILED!")
        print("   Run the diagnostic script to identify the issue:")
        print("   python neural_diagnostic.py")

if __name__ == "__main__":
    main()