from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor
from config.settings_config import get_settings

SUPERVISOR_NAME = "supervisor"

# prompt
supervisor_prompt = """
"""


async def build_supervisor_agent(
    store, checkpointer
) -> tuple[CompiledStateGraph, dict[str, list[str]]]:
    # model
    model = ChatOllama(
        model=get_settings().supervisor_agent_model,  # type: ignore
        temperature=0,  # type: ignore
    )

    supervisor = create_supervisor(
        agents=[],
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
