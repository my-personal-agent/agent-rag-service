from langchain_ollama import OllamaEmbeddings

from config.settings_config import get_settings


def get_lang_store_embeddings() -> tuple[OllamaEmbeddings, int]:
    embeddings = OllamaEmbeddings(
        model=get_settings().lang_store_embeddings_model,
        base_url=str(get_settings().ollama_base_url),
    )

    return embeddings, get_settings().lang_store_embeddings_model_dims
