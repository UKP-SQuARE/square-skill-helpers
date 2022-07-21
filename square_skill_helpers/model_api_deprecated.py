import base64
import logging
import os
from io import BytesIO
from typing import Dict, Iterable

import requests
import numpy as np

from square_skill_helpers import SquareAPI, client_credentials


logger = logging.getLogger(__name__)

class ModelAPIDeprecated(SquareAPI):
    def decode_model_api_response(self, model_api_response: Dict) -> Dict:
        """Decode (if necessary) the model output of the Model API response and make
        it intonumpy arrays.

        Args:
            model_api_response (Dict): The response from the API

        Raises:
            ValueError: Raises ValueError when unexpected types (not `str` or
            `Iterbale`) are provided.

        Returns:
            Dict: model_api_response with 'model_outputs' decoded and parsed to numpy
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
                    f"Encountered unexpected value {type(val)} while trying to decode "
                    f"the model output of the model API. Expected str or iterable."
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

    async def __call__(
        self, model_name: str, pipeline: str, model_request: Dict
    ) -> Dict:
        """Calls the respective pipeline of the Model API. The 'model_outputs' will be 
        decoded if necessary and made into np arrays before returning.
        :param model_request: the request to use for the call
        :return: The response from the Model API. If the request was not succesfull, an exception is raised.

        Args:
            model_name (str): Name of the model to call, e.g. `bert-base-uncased`.
            pipeline (str): Name of the pipeline to use: `embedding`, `generation`,
            `sequence-classification`, `token-classification`, `question-answering`
            model_request (Dict): Dictionary containing parameters that will be sent to
            the model api.

        Raises:
            RuntimeError: Raises RuntimeError, when Model API does not return a status
            code of 200.

        Returns:
            Dict: Model API response.
        """
        url = f"{self.square_api_url}/{model_name}/{pipeline}"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {client_credentials()}"},
            json=model_request,
            verify=os.getenv("VERIFY_SSL", 1) == 1,
        )
        if response.status_code == 200:
            return self.decode_model_api_response(response.json())
        else:
            raise RuntimeError(
                f"Request to model API at URL {url} with request {model_request} "
                f"failed with code {response.status_code} and message {response.text}"
            )
