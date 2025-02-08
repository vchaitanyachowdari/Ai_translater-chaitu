from setuptools import setup, find_packages

setup(
    name="ai_translator",
    version="0.1.0",
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        "torch",
        "transformers",
        "torchaudio",
        "fastapi",
        "pydantic",
        "gtts",
        "soundfile",
    ],
) 