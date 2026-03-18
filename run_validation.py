"""
运行验证流水线的简单脚本
Simple script to run the validation pipeline
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from validation_pipeline import ValidationPipeline

if __name__ == '__main__':
    config_path = 'validation_config_example.yaml'
    
    print("=" * 80)
    print("运行验证流水线 (Running Validation Pipeline)")
    print("=" * 80)
    print(f"\n配置文件: {config_path}\n")
    
    try:
        # 创建并运行验证流水线
        pipeline = ValidationPipeline(config_path)
        pipeline.run_validation()
        
        print("\n" + "=" * 80)
        print("✓ 验证完成! (Validation Complete!)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 错误 (Error): {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
