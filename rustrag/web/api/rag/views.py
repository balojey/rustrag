import os
from typing import Union

from fastapi import APIRouter
import pymongo
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.response import Response

from rustrag.web.api.rag.schema import Message
from settings import settings

router = APIRouter()

os.environ['LLAMA_INDEX_CACHE_DIR'] = os.path.join(os.path.abspath('../'), 'cache')
mongodb_client = pymongo.MongoClient(settings.atlas_uri)
embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
Settings.embed_model = embed_model
llm = LlamaAPI(api_key=settings.llama_api_key)
Settings.llm = llm
vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,
                                 db_name = settings.db_name, collection_name = settings.collection_name,
                                 vector_index_name  = settings.idx_name,
                                 )
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=("""
        You are a helpful rust documentation assistant!
        Your response is always concise, precise, insightful, important and short
    """)
)


@router.post("/")
async def send_message(
    incoming_message: Message,
) -> str:
    """
    Sends response back to user.

    :param incoming_message: incoming message.
    :returns: response to incoming message.
    """
    response = chat_engine.chat(incoming_message.message)
    return response.response
