# -*- coding: utf-8 -*-
"""Embedding模型模块 - 本地embedding模型管理"""

from .embedding_config import SmartLocalEmbedding, create_embedding_model_simple
from .model_utils import check_model_availability_simple

# 简单的embedding模型获取函数
def get_embedding_model_simple():
    """获取简化版embedding模型"""
    return SmartLocalEmbedding(
        model_name="local-embedding-simple",
        dimensions=1536
    )

__all__ = [
    'SmartLocalEmbedding',
    'create_embedding_model_simple',
    'get_embedding_model_simple',
    'check_model_availability_simple'
]
