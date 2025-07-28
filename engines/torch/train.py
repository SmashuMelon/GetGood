#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn
from tqdm import tqdm
import pickle

# Import local modules
from auxiliary_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel


def load_pgn(file_path):
    """Load games from a single PGN file"""
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def load_all_games(data_dir, limit_files=28):
    """Load games from all PGN files in data directory"""
    files = [file for file in os.listdir(data_dir) if file.endswith(".pgn")]
    limit_files = min(len(files), limit_files)
    
    games = []
    print(f"Loading games from {limit_files} PGN files...")
    
    for i, file in enumerate(tqdm(files[:limit_files])):
        games.extend(load_pgn(os.path.join(data_dir, file)))
    
    print(f"GAMES PARSED: {len(games)}")
    return games


def prepare_training_data(games, max_samples=2500000):
    """Convert games to training data"""
    print("Creating training data from games...")
    X, y = create_input_for_nn(games)
    print(f"NUMBER OF SAMPLES: {len(y)}")
    
    # Limit samples if specified
    if max_samples and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Encode moves
    y_encoded, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    return X_tensor, y_tensor, move_to_int, num_classes


def train_model(X, y, num_classes, num_epochs=50, batch_size=64, learning_rate=0.0001, 
                model_save_path="../../models/TORCH_100EPOCHS.pth",
                move_mapping_path="../../models/move_to_int"):
    """Train the chess model"""
    
    # Create Dataset and DataLoader
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Model Initialization
    model = ChessModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time) - minutes * 60
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {minutes}m{seconds}s')
    
    # Save the model
    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Chess Engine')
    parser.add_argument('--data-dir', default='../../data/pgn', 
                       help='Directory containing PGN files')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, 
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=2500000, 
                       help='Maximum number of training samples')
    parser.add_argument('--limit-files', type=int, default=28, 
                       help='Maximum number of PGN files to process')
    parser.add_argument('--model-path', default='../../models/TORCH_100EPOCHS.pth', 
                       help='Path to save trained model')
    parser.add_argument('--mapping-path', default='../../models/move_to_int', 
                       help='Path to save move mapping')
    
    args = parser.parse_args()
    
    try:
        # Load games
        games = load_all_games(args.data_dir, args.limit_files)
        
        # Prepare training data
        X, y, move_to_int, num_classes = prepare_training_data(games, args.max_samples)
        
        # Train model
        model = train_model(X, y, num_classes, 
                          num_epochs=args.epochs,
                          batch_size=args.batch_size,
                          learning_rate=args.learning_rate,
                          model_save_path=args.model_path,
                          move_mapping_path=args.mapping_path)
        
        # Save move mapping
        print(f"Saving move mapping to {args.mapping_path}")
        with open(args.mapping_path, "wb") as file:
            pickle.dump(move_to_int, file)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()