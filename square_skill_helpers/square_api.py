import base64
import logging
from io import BytesIO
from typing import Iterable

import requests
import numpy as np

from square_skill_helpers.config import SquareSkillHelpersConfig

logger = logging.getLogger(__name__)


class SquareAPI:
    def __init__(self, config: SquareSkillHelpersConfig) -> None:
        self.config = config


class ModelAPI(SquareAPI):
    def decode_model_api_response(self, model_api_response):
        """
        Decode (if necessary) the model output of the Model API response and make it into
        np arrays.
        :param model_api_response: The response from the API
        :return: model_api_response with 'model_outputs' decoded and parsed to numpy
        """
        # Decode byte base64 string back to numpy array
        def _decode(arr_string_b64):
            arr_binary_b64 = arr_string_b64.encode()
            arr_binary = base64.decodebytes(arr_binary_b64)
            arr = np.load(BytesIO(arr_binary))
            return arr

        # Recursively go through a value and decodeleaves (=str) or iterate over values and decode them
        def dec_or_iterate(val):
            if isinstance(val, str):
                return _decode(val)
            elif isinstance(val, Iterable):
                return [dec_or_iterate(v) for v in val]
            else:
                raise ValueError(
                    f"Encountered unexpected value {type(val)} while trying to decode the model output of the model API. "
                    f"Expected str or iterable."
                )

        if model_api_response["model_output_is_encoded"]:
            model_api_response["model_outputs"] = {
                key: dec_or_iterate(arr)
                for key, arr in model_api_response["model_outputs"].items()
            }
        else:
            model_api_response["model_outputs"] = {
                key: np.array(arr)
                for key, arr in model_api_response["model_outputs"].items()
            }
        return model_api_response

    async def __call__(self, model_name: str, pipeline: str, model_request):
        """
        Call the Model API with the given complete URL (http://<host>:<port>/api/<model_name>/<endpoint>) and request.
        The 'model_outputs' will be decoded if necessary and made into np arrays before returning.
        :param url: The complete URL to send the request to
        :param model_request: the request to use for the call
        :return: The response from the Model API. If the request was not succesfull, an exception is raised.
        """
        url = f"{self.config.square_api_url}/{model_name}/{pipeline}"
        response = requests.post(
            url,
            json=model_request,
        )
        if response.status_code == 200:
            return self.decode_model_api_response(response.json())
        else:
            raise RuntimeError(
                f"Request to model API at URL {url} with request {model_request} "
                f"failed with code {response.status_code} and message {response.text}"
            )


class DataAPI(SquareAPI):
    async def __call__(
        self, datastore_name: str, index_name: str, query: str, top_k: int = 10
    ):
        """
        Call /datastores/{datstore_name}/search
        """
        url = f"{self.config.square_api_url}/datastores/{datastore_name}/search"
        response = requests.get(
            url,
            params=dict(index_name=index_name, query=query, top_k=top_k),
            headers={"Authorization": self.config.data_api_key},
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
