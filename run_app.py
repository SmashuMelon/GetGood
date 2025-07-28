#!/usr/bin/env python3
"""
Chess Engine API Runner
Starts the Flask API server for the chess engine application.
"""

import os
import sys
import argparse
from pathlib import Path

def setup_environment():
    """Setup the environment for running the Flask app"""
    # Get the directory where this script is located (should be pilot-chess root)
    current_dir = Path(__file__).parent.absolute()
    
    # Add the current directory to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add engines/torch to path
    engines_torch_path = current_dir / 'engines' / 'torch'
    if str(engines_torch_path) not in sys.path:
        sys.path.insert(0, str(engines_torch_path))
    
    print(f"Working directory: {current_dir}")
    print(f"Python path includes: {engines_torch_path}")
    
    return current_dir

def check_dependencies():
    """Check if required modules can be imported"""
    print("Checking dependencies...")
    
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError as e:
        print(f"✗ Flask not found: {e}")
        return False
    
    try:
        import chess
        print(f"✓ python-chess {chess.__version__}")
    except ImportError as e:
        print(f"✗ python-chess not found: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    # Check if our custom modules can be imported
    try:
        from engines.torch.predict import ChessPredictor
        print("✓ ChessPredictor module")
    except ImportError as e:
        print(f"✗ ChessPredictor module: {e}")
        return False
    
    try:
        from engines.torch.alphabeta_engine import AlphaBetaEngine
        print("✓ AlphaBetaEngine module")
    except ImportError as e:
        print(f"✗ AlphaBetaEngine module: {e}")
        return False
    
    return True

def check_model_files(models_dir):
    """Check if model files exist"""
    print(f"\nChecking model files in {models_dir}...")
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return False
    
    # Look for common model file patterns
    model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
    mapping_files = list(models_dir.glob("*mapping*")) + list(models_dir.glob("*.json"))
    
    print(f"Found model files: {[f.name for f in model_files]}")
    print(f"Found mapping files: {[f.name for f in mapping_files]}")
    
    if not model_files:
        print("⚠ No .pth or .pt model files found")
    if not mapping_files:
        print("⚠ No mapping files found")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the Chess Engine API server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--neural-model', help='Path to neural network model file')
    parser.add_argument('--neural-mapping', help='Path to neural network mapping file')
    parser.add_argument('--alphabeta-depth', type=int, default=4, help='Alpha-beta search depth (default: 4)')
    parser.add_argument('--stockfish-path', help='Path to Stockfish executable')
    parser.add_argument('--neural-weight', type=float, default=0.4, help='Weight for neural network in hybrid engine (default: 0.4)')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and file checks')
    
    args = parser.parse_args()
    
    # Setup environment
    current_dir = setup_environment()
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            print("\n❌ Dependency check failed. Please install missing packages:")
            print("pip install -r requirements.txt")
            return 1
        
        # Check model files
        models_dir = current_dir / 'models'
        check_model_files(models_dir)
    
    print(f"\n🚀 Starting Chess Engine API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    
    try:
        # Import the Flask app
        from app import app, engine_manager
        
        # Initialize engines with custom parameters if provided
        if any([args.neural_model, args.neural_mapping, args.stockfish_path]) or args.alphabeta_depth != 4 or args.neural_weight != 0.4:
            print("Reinitializing engines with custom parameters...")
            engine_manager.initialize_engines(
                neural_model_path=args.neural_model,
                neural_mapping_path=args.neural_mapping,
                alphabeta_depth=args.alphabeta_depth,
                stockfish_path=args.stockfish_path,
                neural_weight=args.neural_weight
            )
        
        # Start the Flask app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("Make sure you're running this from the pilot-chess root directory")
        return 1
    except KeyboardInterrupt:
        print("\n⏹ Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    finally:
        # Cleanup
        try:
            if 'engine_manager' in locals():
                engine_manager.cleanup()
        except:
            pass

if __name__ == '__main__':
    sys.exit(main())