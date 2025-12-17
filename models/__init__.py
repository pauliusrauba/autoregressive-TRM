# models/__init__.py
from .gpt import GPTBase
from .gpt_level1 import GPTLevel1
from .gpt_level2 import GPTLevel2
from .ut import UT
#from .ut_gpt import LitUnifiedGPTUT

model_registry = {
    "gpt": GPTBase,
    "gpt_level1": GPTLevel1,
    "gpt_level2": GPTLevel2,
    "ut": UT
}

def build_model(name: str, **kwargs):
    ModelClass = model_registry[name]
    return ModelClass(**kwargs)
