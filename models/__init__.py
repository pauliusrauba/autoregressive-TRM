# models/__init__.py
from .gpt import LitGPT


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "gpt":
        return LitGPT(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
