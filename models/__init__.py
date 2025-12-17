# models/__init__.py
from .gpt import GPTBase
from .gpt_level1 import GPTLevel1
from .gpt_level2 import GPTLevel2
from .ut import UT
#from .ut_gpt import LitUnifiedGPTUT


def build_model(name: str, **kwargs):
    if name == "gpt":
        return GPTBase(**kwargs)
    elif name == "gpt_level1":
        return GPTLevel1(**kwargs)
    elif name == "gpt_level2":
        return GPTLevel2(**kwargs)
    elif name == "gpt_level3" or name == "ut":
        return UT(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")