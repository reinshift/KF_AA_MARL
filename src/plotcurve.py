import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_rewards(file_path, window_size=5, save_plot=False, output_path=None):
    """
    Plot reward curves
    
    Args:
        file_path: Path to CSV file
        window_size: Smoothing window size
        save_plot: Whether to save the plot
        output_path: Output path for the saved plot
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Read data
    try:
        data = pd.read_csv(file_path)
        episodes = data['episode']
        total_reward_hunters = data['total_reward_hunters']
        total_reward_targets = data['total_reward_targets']
        
        # Apply moving average smoothing
        if window_size > 1:
            total_reward_hunters_smooth = total_reward_hunters.rolling(window=window_size).mean()
            total_reward_targets_smooth = total_reward_targets.rolling(window=window_size).mean()
        else:
            total_reward_hunters_smooth = total_reward_hunters
            total_reward_targets_smooth = total_reward_targets
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot hunters rewards
        plt.subplot(2, 1, 1)
        plt.plot(episodes, total_reward_hunters, 'b-', alpha=0.3, label='Raw Data')
        plt.plot(episodes, total_reward_hunters_smooth, 'b-', label=f'Hunters Reward (window={window_size})')
        plt.title('Hunters Total Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Hunters Total Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot targets rewards
        plt.subplot(2, 1, 2)
        plt.plot(episodes, total_reward_targets, 'r-', alpha=0.3, label='Raw Data')
        plt.plot(episodes, total_reward_targets_smooth, 'r-', label=f'Targets Reward (window={window_size})')
        plt.title('Targets Total Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Targets Total Reward')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if save_plot and output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        raise Exception(f"Error reading or processing CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot training reward curves')
    parser.add_argument('--file_path', type=str, default=None, help='Path to CSV file')
    parser.add_argument('--window_size', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--save_plot', action='store_true', help='Save the plot')
    parser.add_argument('--output_path', type=str, default=None, help='Output path for the plot')
    
    args = parser.parse_args()
    
    # If no file path specified, use default path
    if args.file_path is None:
        # Try to find the latest training data
        data_train_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_train')
        if not os.path.exists(data_train_dir):
            print("Error: Training data directory not found. Please train first or specify a CSV file path.")
            return
            
        # Find the latest folder
        folders = [os.path.join(data_train_dir, f) for f in os.listdir(data_train_dir) if os.path.isdir(os.path.join(data_train_dir, f))]
        if not folders:
            print("Error: Training data directory is empty. Please train first or specify a CSV file path.")
            return
            
        latest_folder = max(folders, key=os.path.getmtime)
        csv_path = os.path.join(latest_folder, 'rewards.csv')
        
        if not os.path.exists(csv_path):
            print(f"Error: Could not find rewards.csv in the latest training directory {latest_folder}. Please train first or specify a CSV file path.")
            return
            
        args.file_path = csv_path
        print(f"Using latest training data: {args.file_path}")
    
    try:
        plot_rewards(args.file_path, args.window_size, args.save_plot, args.output_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()