import dataclasses
import os
import warnings
from dataclasses import dataclass

from dotenv import load_dotenv

@dataclass
class SquareSkillHelpersConfig:
    model_api_user: str
    model_api_password: str
    model_api_url: str = "http://traefik:80/api"
    data_api_key: str
    data_api_url: str = "http://datastore_api:7000"

    @classmethod
    def from_dotenv(cls, fp: str = None):
        if fp is not None:
            if not os.path.exists(fp):
                warnings.warn(
                    (
                        f"No env file found at {fp}. "
                        f"Attempting to load from existing env variables."
                    ),
                    RuntimeWarning
                )
            else:
                load_dotenv(fp)

        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name.upper() in os.environ:
                kwargs[field.name] = os.environ[field.name.upper()]
        return cls(**kwargs)
