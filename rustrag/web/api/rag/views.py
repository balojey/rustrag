import os
from fastapi import APIRouter
import pymongo
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.llms.llama_api import LlamaAPI

from rustrag.web.api.rag.schema import Message
from settings import settings

router = APIRouter()

ATLAS_URI = os.getenv("ATLAS_URI")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

os.environ['LLAMA_INDEX_CACHE_DIR'] = os.path.join(os.path.abspath('../'), 'cache')
mongodb_client = pymongo.MongoClient(ATLAS_URI)
embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
Settings.embed_model = embed_model
llm = LlamaAPI(api_key=LLAMA_API_KEY)
Settings.llm = llm
vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,
                                 db_name = settings.db_name, collection_name = settings.collection_name,
                                 index_name  = settings.idx_name,
                                 )
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)


@router.post("/")
async def send_message(
    incoming_message: Message,
) -> Message:
    """
    Sends response back to user.

    :param incoming_message: incoming message.
    :returns: response to incoming message.
    """
    outgoing_message = Message(message=index.as_query_engine().query(incoming_message.message))
    return outgoing_message
