import dataclasses
import os
import warnings
from dataclasses import dataclass

from dotenv import load_dotenv

@dataclass
class SquareSkillHelpersConfig:
    data_api_key: str
    square_api_url: str = "https://square.ukp-lab.de/api"

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
