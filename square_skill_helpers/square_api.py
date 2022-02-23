import os


class SquareAPI():
    def __init__(self) -> None:
        self.square_api_url = os.getenv("SQUARE_API_URL")
