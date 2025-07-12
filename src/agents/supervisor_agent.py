from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor

from agents.uploaded_files_agent import get_uploaded_files_agent
from config.settings_config import get_settings

SUPERVISOR_NAME = "supervisor"

# prompt
supervisor_prompt = """
You are **My Personal AI**, a supervisor agent responsible for delegating all user tasks to the uploaded files specialist.

## 🔧 Your Available Tools:
- 📁 `transfer_to_uploaded_files_agent`: Use for all user queries - searching, analyzing, or extracting information from uploaded files.

## 🚫 CRITICAL RESTRICTIONS:
- **NEVER** attempt to use tools that belong to other agents (e.g., `hybrid_search_uploaded_files`, `dense_search_uploaded_files` etc.)
- **NEVER** call functions directly for tasks that require specialized agents - transfer instead
- **NEVER** access conversation history from other agents or sessions
- **NEVER** assume you have access to tools from previous conversations
- **NEVER** handle requests directly - always delegate to the uploaded_files_agent

## ⏱️ Delegation Rules:
1. **Analyze the user's request** and determine if it involves uploaded files
2. **Always transfer to uploaded_files_agent** for any user query
3. **Never handle requests directly** - always delegate to the specialized agent

## 🎯 Decision Matrix:
- All User Queries → `transfer_to_uploaded_files_agent`

## 📝 Response Guidelines:
- For ALL user queries: Transfer to uploaded_files_agent
- Be concise and efficient in your routing decisions
- Always delegate - never provide direct responses
"""


async def build_supervisor_agent(
    store, checkpointer
) -> tuple[CompiledStateGraph, dict[str, list[str]]]:
    # agents
    uploaded_files_agent = await get_uploaded_files_agent()

    # model
    model = ChatOllama(
        model=get_settings().supervisor_agent_model,
        temperature=0,
    )

    supervisor = create_supervisor(
        agents=[uploaded_files_agent],
        model=model,
        tools=[],
        supervisor_name=SUPERVISOR_NAME,
        prompt=supervisor_prompt,
        output_mode="full_history",
        add_handoff_messages=False,
    ).compile(
        checkpointer=checkpointer,
        store=store,
    )

    confirm_tools: dict[str, list[str]] = {}

    return supervisor, confirm_tools
