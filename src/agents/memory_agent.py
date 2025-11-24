from typing import Dict, List, Optional, Any
import asyncio
import json
import os
import hashlib
import math
import random
from datetime import datetime
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.memory import InMemoryMemory, Mem0LongTermMemory
from agentscope.model import OpenAIChatModel
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse, EmbeddingUsage
from agentscope.formatter import DeepSeekChatFormatter
from pydantic import BaseModel, Field

from ..memory.skill_manager import SkillManager
from ..memory.extractor import SkillExtractor


class NeedMemory(BaseModel):
    need_long_term: bool = Field(default=False, description="Need long term memory")


class SmartLocalEmbedding(EmbeddingModelBase):
    """Smart local embedding system, prioritizes pre-downloaded models, no network needed"""
    
    supported_modalities = ["text"]
    
    def __init__(self, model_name: str = "local-smart", dimensions: int = 1536) -> None:
        super().__init__(model_name, dimensions)
        self.model = None
        self.model_path = "models/embedding_model"
        self.model_status = "unknown"
        self._initialize_model()
    
    def _initialize_model(self):
        """Smart model initialization"""
        print("Initializing smart local embedding system...")
        
        if self._check_local_model():
            self.model_status = "local_model"
            print("Detected local pre-downloaded model, using local model")
        else:
            print("No local model detected, download first.")
    
    def _check_local_model(self):
        """Check for local pre-downloaded model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_path = Path(self.model_path)
            print(f"Checking model path: {model_path.absolute()}")
            
            model_files = [
                "config.json",
                "model.safetensors", 
                "pytorch_model.bin",
                "modules.json"
            ]
            
            has_model = any((model_path / file).exists() for file in model_files)
            
            if has_model:
                print(f"Loading local model: {self.model_path}")
                self.model = SentenceTransformer(str(model_path))
                print(f"Successfully loaded local model: {self.model_path}")
                return True
            else:
                print(f"No local model files found")
                print(f"Directory contents: {[f.name for f in model_path.iterdir() if f.is_file()]}")
                return False
        except ImportError:
            print("sentence-transformers not installed, using pure mathematical method")
            return False
        except Exception as e:
            print(f"Failed to load local model: {e}, using mathematical method")
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
        
        if self.model is not None:
            try:
                start_time = datetime.now()
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.encode, gather_text
                )
                time = (datetime.now() - start_time).total_seconds()
                
                embeddings = self._adjust_dimensions(embeddings.tolist())
                
                return EmbeddingResponse(
                    embeddings=embeddings,
                    usage=EmbeddingUsage(tokens=sum(len(t) for t in gather_text), time=time),
                    source="local_model"
                )
            except Exception as e:
                print(f"Local model call failed: {e}")
    
    def _adjust_dimensions(self, embeddings):
        """Adjust vector dimensions to target dimension"""
        adjusted = []
        for emb in embeddings:
            current_dim = len(emb)
            if current_dim < self.dimensions:
                extended = self._extend_vector(emb, self.dimensions)
                adjusted.append(extended)
            elif current_dim > self.dimensions:
                truncated = emb[:self.dimensions]
                norm = sum(x*x for x in truncated) ** 0.5
                normalized = [x/norm for x in truncated]
                adjusted.append(normalized)
            else:
                adjusted.append(emb)
        return adjusted
    
    def _extend_vector(self, vector, target_dim):
        """Extend vector using mathematical method"""
        extended = vector.copy()
        current_len = len(extended)
        
        seed = sum(ord(c) for c in str(extended)) % 1000
        rng = random.Random(seed)
        
        while len(extended) < target_dim:
            for val in vector:
                if len(extended) >= target_dim:
                    break
                transformed = math.sin(val * (len(extended) + 1) + seed) * 0.5 + 0.5
                extended.append(transformed)
        
        norm = sum(x*x for x in extended) ** 0.5
        return [x/norm for x in extended]

def create_qwen_model():
    return OpenAIChatModel(
        model_name="Qwen3-235B-A22B",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        client_args={"base_url": "https://ai.api.coregpu.cn/v1"},
        stream=False,
    )


def create_embedding_model():
    return SmartLocalEmbedding(
        model_name="local-smart",
        dimensions=1536
    )


# Simplified long-term memory without Qdrant - using local file storage
_long_term_memory_cache = {}

async def create_long_term_memory(agent: ReActAgent, agent_name: str, user_name: str = "default_user"):
    """
    Configure long-term memory for existing ReActAgent using local file storage
    Simplified for single agent without Qdrant dependency
    """
    cache_key = f"{agent_name}_{user_name}"
    
    if cache_key in _long_term_memory_cache:
        print(f"Using cached long-term memory: {agent_name}")
        memory = _long_term_memory_cache[cache_key]
    else:
        print(f"Creating new long-term memory: {agent_name}")
        
        # Use local file storage instead of Qdrant
        storage_path = f"./mem0_data/{agent_name}_{user_name}"
        os.makedirs(storage_path, exist_ok=True)
        
        from mem0.vector_stores.configs import VectorStoreConfig
        
        vector_store_config = VectorStoreConfig(
            provider='qdrant',
            config={
                'path': storage_path,
                'on_disk': True,
                'collection_name': f'{agent_name}_{user_name}'
            }
        )
        
        memory = Mem0LongTermMemory(
            agent_name=agent_name,
            user_name=user_name,
            model=create_qwen_model(),
            embedding_model=create_embedding_model(),
            vector_store_config=vector_store_config
        )
        
        _long_term_memory_cache[cache_key] = memory
    
    agent.memory = memory
    return agent


class MemoryAgent(ReActAgent):
    def __init__(
        self,
        name: str,
        model_config: Dict[str, Any],
        skill_manager: SkillManager,
        system_prompt: str = "",
        memory: Optional[InMemoryMemory] = None,
        verbose: bool = True,
        user_name: str = "default_user",
        enable_long_term_memory: bool = True,
        **kwargs
    ):
        # Create model from model_config
        model = OpenAIChatModel(
            model_name=model_config.get("model_name", "DeepSeek-R1"),
            api_key=model_config.get("api_key", os.environ.get("DEEPSEEK_API_KEY", "")),
            client_args=model_config.get("client_args", {"base_url": "https://ai.api.coregpu.cn/v1"}),
            stream=model_config.get("stream", False),
        )
        
        # Create formatter
        formatter = DeepSeekChatFormatter()
        
        super().__init__(
            name=name,
            model=model,
            sys_prompt=system_prompt,
            formatter=formatter,
            memory=memory or InMemoryMemory(),
            **kwargs
        )
        
        self.skill_manager = skill_manager
        self.skill_extractor = SkillExtractor(model_config)
        self.episode_trace = []
        self.current_task = None
        self.user_name = user_name
        self.enable_long_term_memory = enable_long_term_memory
        self.memory_type = "short_term"
        self.verbose = verbose
    
    async def __call__(self, msg: Msg) -> Msg:
        task_description = msg.content
        self.current_task = task_description
        
        relevant_skills = await self._retrieve_relevant_skills(task_description)
        relevant_memories = await self._retrieve_relevant_memories(task_description)
        
        skills_prompt = self._format_skills_for_prompt(relevant_skills)
        memories_prompt = self._format_memories_for_prompt(relevant_memories)
        
        enhanced_prompt = f"""
        Current task: {task_description}
        
        {skills_prompt}
        
        {memories_prompt}
        
        Please develop action plan based on above information.
        """
        
        enhanced_msg = Msg(
            name=msg.name,
            content=enhanced_prompt,
            role=msg.role
        )
        
        response = await super().__call__(enhanced_msg)
        
        self._record_interaction(task_description, enhanced_prompt, response)
        
        return response
