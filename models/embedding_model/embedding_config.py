# -*- coding: utf-8 -*-
"""Embedding配置 - 本地embedding模型实现"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse, EmbeddingUsage
from agentscope.message import TextBlock
import hashlib
import math
import random

class SmartLocalEmbedding(EmbeddingModelBase):
    """智能本地embedding系统，优先使用预下载模型，无需网络"""
    
    supported_modalities = ["text"]
    
    def __init__(self, model_name: str = "local-smart", dimensions: int = 1536) -> None:
        super().__init__(model_name, dimensions)
        self.model = None
        self.model_path = "A2S/models/embedding_model"  # 指向新的embedding模型路径
        self.model_status = "unknown"
        self._initialize_model()
    
    def _initialize_model(self):
        """智能初始化模型"""
        print("🔍 正在初始化智能本地embedding系统...")
        
        # 检查本地模型
        if self._check_local_model():
            self.model_status = "local_model"
            print("✅ 检测到本地预下载模型，将使用本地模型")
        else:
            self.model_status = "mathematical"
            print("⚠️  未检测到本地模型，将使用数学方法生成embedding")
    
    def _check_local_model(self):
        """检查本地预下载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_path = Path(self.model_path)
            print(f"🔍 检查embedding模型路径: {model_path.absolute()}")
            
            # 检查embedding模型文件
            model_files = [
                "config.json",
                "model.safetensors", 
                "pytorch_model.bin",
                "modules.json",
                "tokenizer.json"  # 添加tokenizer检查
            ]
            
            # 检查是否有任何模型文件
            has_model = any((model_path / file).exists() for file in model_files)
            
            if has_model:
                print(f"🔄 正在加载本地embedding模型: {self.model_path}")
                self.model = SentenceTransformer(str(model_path))
                print(f"✅ 成功加载本地embedding模型: {self.model_path}")
                return True
            else:
                print(f"📭 未找到本地embedding模型文件")
                if model_path.exists():
                    files = [f.name for f in model_path.iterdir() if f.is_file()]
                    print(f"📂 embedding目录内容: {files}")
                return False
        except ImportError:
            print("⚠️  未安装sentence-transformers，将使用纯数学方法")
            return False
        except Exception as e:
            print(f"⚠️  加载本地embedding模型失败: {e}，将使用数学方法")
            return False
    
    async def __call__(self, text: list[str | TextBlock], **kwargs: any) -> EmbeddingResponse:
        gather_text = []
        for item in text:
            if isinstance(item, dict) and "text" in item:
                gather_text.append(item["text"])
            elif isinstance(item, str):
                gather_text.append(item)
            else:
                raise ValueError("Input text must be a list of strings or TextBlock dicts.")
        
        # 如果有本地模型，优先使用
        if self.model is not None:
            try:
                start_time = datetime.now()
                # 在异步函数中运行同步模型
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.encode, gather_text
                )
                time = (datetime.now() - start_time).total_seconds()
                
                # 调整维度到1536（如果需要）
                embeddings = self._adjust_dimensions(embeddings.tolist())
                
                return EmbeddingResponse(
                    embeddings=embeddings,
                    usage=EmbeddingUsage(tokens=sum(len(t) for t in gather_text), time=time),
                    source="local_model"
                )
            except Exception as e:
                print(f"❌ 本地embedding模型调用失败: {e}，降级到数学方法")
        
        # 降级方案：使用数学方法生成向量
        print("⚠️  使用数学方法生成embedding（完全离线）")
        embeddings = self._generate_mathematical_embeddings(gather_text)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=sum(len(t) for t in gather_text), time=0.01),
            source="mathematical"
        )
    
    def _adjust_dimensions(self, embeddings):
        """调整向量维度到目标维度"""
        adjusted = []
        for emb in embeddings:
            current_dim = len(emb)
            if current_dim < self.dimensions:
                # 扩展维度：使用数学方法扩展
                extended = self._extend_vector(emb, self.dimensions)
                adjusted.append(extended)
            elif current_dim > self.dimensions:
                # 缩减维度：截断并重新归一化
                truncated = emb[:self.dimensions]
                norm = sum(x*x for x in truncated) ** 0.5
                normalized = [x/norm for x in truncated]
                adjusted.append(normalized)
            else:
                adjusted.append(emb)
        return adjusted
    
    def _extend_vector(self, vector, target_dim):
        """使用数学方法扩展向量"""
        extended = vector.copy()
        current_len = len(extended)
        
        # 使用不同的数学变换扩展
        seed = sum(ord(c) for c in str(extended)) % 1000
        rng = random.Random(seed)
        
        while len(extended) < target_dim:
            # 基于现有向量的数学变换
            for val in vector:
                if len(extended) >= target_dim:
                    break
                # 添加一些变化但保持确定性
                transformed = math.sin(val * (len(extended) + 1) + seed) * 0.5 + 0.5
                extended.append(transformed)
        
        # 归一化
        norm = sum(x*x for x in extended) ** 0.5
        return [x/norm for x in extended]
    
    def _generate_mathematical_embeddings(self, texts):
        """使用数学方法生成embedding向量"""
        embeddings = []
        for text in texts:
            # 基于文本内容的确定性向量生成
            hash_val = hashlib.md5(text.encode()).hexdigest()
            
            # 生成基础向量
            base_vector = []
            for i in range(0, 32, 2):  # MD5有32个十六进制字符
                # 将每两个字符转换为0-1之间的浮点数
                val = int(hash_val[i:i+2], 16) / 255.0
                base_vector.append(val)
            
            # 扩展到目标维度
            full_vector = []
            seed = hash(text) % 1000
            rng = random.Random(seed)
            
            while len(full_vector) < self.dimensions:
                # 使用不同的数学变换扩展
                for val in base_vector:
                    if len(full_vector) >= self.dimensions:
                        break
                    # 添加一些变化但保持确定性
                    transformed = math.sin(val * (len(full_vector) + 1) + seed) * 0.5 + 0.5
                    full_vector.append(transformed)
            
            # 归一化
            norm = sum(x*x for x in full_vector) ** 0.5
            normalized = [x/norm for x in full_vector]
            embeddings.append(normalized[:self.dimensions])
        
        return embeddings

def create_embedding_model_simple():
    """创建简化版embedding模型"""
    return SmartLocalEmbedding(
        model_name="local-embedding-simple",
        dimensions=1536
    )
