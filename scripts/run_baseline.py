import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.models import OpenAIChatModel, DashScopeChatModel

from src.adapters.alfworld_adapter import AlfworldAdapter

async def main():
    print("Initializing ALFWorld Environment...")
    adapter = AlfworldAdapter()
    
    toolkit = Toolkit()
    tool_cfg = adapter.get_tool_config()
    toolkit.register_tool_function(
        tool_cfg["function"],
        function_name=tool_cfg["function_name"],
        function_desc=tool_cfg["function_desc"]
    )

    model = OpenAIChatModel(
        model_name="DeepSeek-R1",
        # model_name="Qwen3-235B-A22B",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        client_args={
            "base_url": "https://ai.api.coregpu.cn/v1"
        },
        stream=False,
    )

    sys_prompt = """
    You are an embodied intelligent assistant.
    Your task is to use the tool 'interact_with_environment' to complete tasks in the simulated household environment.
    
    Rules:
    1. You must send specific commands through the tool (e.g., 'go to sink 1', 'take apple').
    2. Carefully observe the Observation returned by the tool.
    3. If you observe task completion (SUCCESS), please stop.
    """

    agent = ReActAgent(
        name="BaselineAgent",
        sys_prompt=sys_prompt,
        model=model,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        verbose=True
    )

    print("Resetting Environment...")
    task_desc = adapter.reset()
    print(f"Current Task: {task_desc}")

    msg = Msg(name="system", content=f"Your Goal: {task_desc}", role="user")
    
    await agent(msg)

if __name__ == "__main__":
    asyncio.run(main())
