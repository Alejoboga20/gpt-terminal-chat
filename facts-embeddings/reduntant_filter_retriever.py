from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str):
        # Caclulate embeddings for the query
        query_embeddings = self.embeddings.embed_query(query)
        # Take embeddings and feed them into that max_margianl_relevance_search_by_vector
        results = self.chroma.max_marginal_relevance_search_by_vector(
            embedding=query_embeddings, lambda_mult=0.8)

        return results

    async def get_relevant_documents(self):
        return []
