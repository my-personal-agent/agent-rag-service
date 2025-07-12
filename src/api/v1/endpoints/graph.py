import logging

from fastapi import APIRouter, Request
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/invoke")
async def invoke_graph(
    request: Request, payload: MessagesState, config: RunnableConfig
):
    supervisor_agent: CompiledStateGraph = request.app.state.supervisor_agent
    result = await supervisor_agent.ainvoke(payload, config=config)
    return result


@router.post("/stream")
async def stream_graph(
    request: Request, payload: MessagesState, config: RunnableConfig
):
    supervisor_agent: CompiledStateGraph = request.app.state.supervisor_agent
    return supervisor_agent.astream(payload, config=config)
