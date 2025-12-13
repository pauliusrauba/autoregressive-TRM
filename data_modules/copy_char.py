from .algorithmic_char import load_algorithmic_char

def load_copy_char(**kwargs):
    return load_algorithmic_char(task="copy", **kwargs)
