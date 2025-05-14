#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    parser = argparse.ArgumentParser(description="KF_AA_MARL")
    parser.add_argument("mode", choices=["train", "train_continue", "test", "plot"], 
                        help="train, train_continue, test, plot")
    parser.add_argument("--file_path", type=str, default=None, 
                        help="Path to CSV file (only for plot mode)")
    args = parser.parse_args()
    
    if sys.platform == "win32":
        # Windows
        script_ext = ".bat"
    else:
        # Linux/Mac
        script_ext = ".sh"
        os.system(f"chmod +x scripts/{args.mode}{script_ext}")
    
    if args.mode == "plot":
        plot_cmd = f"python src/plotcurve.py --window_size 10"
        if args.file_path:
            file_path = args.file_path
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(file_path)
            plot_cmd += f" --file_path \"{file_path}\""
        
        os.system(plot_cmd)
    else:
        if sys.platform == "win32":
            os.system(f"cd scripts && {args.mode}{script_ext}")
        else:
            os.system(f"cd scripts && ./{args.mode}{script_ext}")

if __name__ == "__main__":
    main() 