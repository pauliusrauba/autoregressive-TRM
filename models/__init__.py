# models/__init__.py
from .gpt import LitGPT
from .ut_gpt import LitUnifiedGPTUT


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "gpt":
        return LitGPT(**kwargs)
    elif name == "ut_gpt":
        return LitUnifiedGPTUT(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
