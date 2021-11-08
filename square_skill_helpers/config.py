import dataclasses
import os
from dataclasses import dataclass

from dotenv import load_dotenv

@dataclass
class SquareSkillHelpersConfig:
    model_api_key: str
    model_api_url: str = "http://model_nginx:8080/api"
    data_api_url: str = "http://host.docker.internal:8002/datastores"

    @classmethod
    def from_dotenv(cls, fp: str = ".env"):
        load_dotenv(fp)

        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name.upper() in os.environ:
                kwargs[field.name] = os.environ[field.name.upper()]
        return cls(**kwargs)