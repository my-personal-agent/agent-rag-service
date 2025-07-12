# Application metrics
from prometheus_client import Counter, Gauge, Histogram, Info

from config.settings_config import get_settings

api_calls_counter = Counter(
    "rag_agent_api_calls_total",
    "Total RAG Agent API calls",
    ["endpoint", "method", "status"],
)
api_duration_histogram = Histogram(
    "rag_agent_api_duration_seconds",
    "RAG Agent API execution time",
    ["endpoint", "method"],
)
active_connections = Gauge(
    "rag_agent_api_active_connections", "Number of active connections"
)
server_info = Info("rag_agent_api_server_info", "Server info")

# System metrics
memory_usage = Gauge("rag_agent_api_memory_usage_bytes", "Memory usage in bytes")
cpu_usage = Gauge("rag_agent_api_cpu_usage_percent", "CPU usage percent")

# Set static metadata for server
server_info.info(
    {
        "version": get_settings().project_version,
        "name": get_settings().project_name,
        "framework": "FastAPI",
    }
)
