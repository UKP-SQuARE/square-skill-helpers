import logging
import os
from typing import Dict, List

import requests

from square_skill_helpers import SquareAPI, client_credentials

logger = logging.getLogger(__name__)


class DataAPI(SquareAPI):
    async def __call__(
        self,
        datastore_name: str,
        query: str,
        index_name: str = None,
        top_k: int = 10,
        feedback_documents: List[str] = None
    ) -> Dict:
        """Calls the /search endpoint of the Datastore API.

        Args:
            datastore_name (str): Name of the datastore to use. E.g. `nq`.
            query (str): The query to be used for searching documents.
            index_name (str, optional): Name of index to be used. If `None`, the
            ElasticSearch index will be used, using BM25 as retrieval method. Defaults
            to `None`.
            top_k (int, optional): [description]. Number of documents to return.
            Defaults to 10.
            feedback_documents (List[str], optional): List of relevant feedback documents from a previous query

        Raises:
            RuntimeError: Raises RuntimeError, when Datastore API does not return a
            status code of 200.

        Returns:
            Dict: Datastore API response.
        """
        url = f"{self.square_api_url}/datastores/{datastore_name}/search"
        response = requests.get(
            url,
            params=dict(index_name=index_name, query=query, top_k=top_k, feedback_documents=feedback_documents),
            headers={"Authorization": f"Bearer {client_credentials()}"},
            verify=os.getenv("VERIFY_SSL", 1) == 1,
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(
                f"Request to data API at URL {url} with request "
                f"datastore_name={datastore_name} index_name={index_name} "
                f"query={query} top_k={top_k} . failed with code "
                f"{response.status_code} and message {response.text}"
            )
