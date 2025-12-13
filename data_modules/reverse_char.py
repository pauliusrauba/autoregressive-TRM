from .algorithmic_char import load_algorithmic_char

def load_reverse_char(**kwargs):
    return load_algorithmic_char(task="reverse", **kwargs)