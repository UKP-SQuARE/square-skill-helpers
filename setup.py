
from setuptools import setup, find_packages

__version__ = "0.0.10"

setup(
    name="square_skill_helpers",
    version=__version__,
    license="MIT",
    description="",
    url="https://github.com/UKP-SQuARE/square-skill-helpers",
    download_url=f"https://github.com/UKP-SQuARE/square-skill-helpers/archive/refs/tags/v{__version__}.tar.gz",
    author="UKP",
    author_email="baumgaertner@ukp.informatik.tu-darmstadt.de",
    packages=find_packages(exclude=(".gitignore", "tests")),
    install_requires=[
        "requests==2.26.0",
        "numpy==1.21.3",
        "square-auth>=0.0.3",
        "aiohttp>=3.8.1"
    ],
)
