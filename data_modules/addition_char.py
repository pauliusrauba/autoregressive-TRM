from .algorithmic_char import load_algorithmic_char

def load_addition_char(**kwargs):
    return load_algorithmic_char(task="addition", **kwargs)
