import os

class SquareAPI():
    def __init__(self) -> None:
        self.square_api_url = os.getenv("SQUARE_API_URL")
        self.verify_ssl = os.getenv("VERIFY_SSL") == "1"
