"""Configuration settings for the transformer model."""
from pathlib import Path

def get_config():
    """Returns the default configuration dictionary for the transformer model."""
    return {
        'batch_size': 16,
        'num_epochs': 25,
        'lr': 10**-4,
        'seq_len': 450,
        'd_model': 512,
        'src_lang': 'en',
        'tgt_lang': 'hi',
        'model_folder': 'weights',
        'model_basename': 'en-hi_',
        'preload': '16',
        'tokenizer_file': "tokenizer_{0}.json",
        'experiment_name': 'runs/tmodel',
        'train_only_split': True,
        'dataset_path': 'cfilt/iitb-english-hindi',
        'dataset_name': None
    }

def get_weights_file_path(config, epoch: str):
    """Generates the file path for saving/loading model weights for a given epoch."""
    model_folder = config['model_folder']
    model_basename =  config['model_basename']
    model_filename =  f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
