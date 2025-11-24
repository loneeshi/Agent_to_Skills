# -*- coding: utf-8 -*-
"""模型工具函数 - 检查模型可用性等工具函数"""

from pathlib import Path

def check_model_availability_simple():
    """简化版模型可用性检查"""
    model_path = Path("A2S/models/embedding_model")
    print(f"🔍 检查embedding模型路径: {model_path.absolute()}")
    
    # 检查基本模型文件
    basic_files = ["config.json", "model.safetensors", "tokenizer.json"]
    has_basic_model = any((model_path / file).exists() for file in basic_files)
    
    if has_basic_model:
        print("✅ 检测到基本embedding模型文件")
        # 列出目录内容供调试
        if model_path.exists():
            files = [f.name for f in model_path.iterdir() if f.is_file()]
            print(f"📂 embedding模型目录内容: {files}")
        return True
    else:
        print("⚠️  未检测到基本embedding模型文件，将使用数学方法")
        if model_path.exists():
            files = [f.name for f in model_path.iterdir() if f.is_file()]
            print(f"📂 目录内容: {files}")
        return False
