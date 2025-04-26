"""Module for training a transformer model for bilingual translation tasks.

This module includes functions for tokenizer creation, dataset preparation, and data loading.
"""
# Standard Library
from pathlib import Path
from torch.utils.data import random_split, DataLoader

# Third-Party Libraries
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# Local Modules
from config import get_config, get_weights_file_path
from dataset import BillingualDataset, causal_mask
from model import build_transformer

def get_all_sentences(ds, lang):
    """Generator function to yield sentences from a dataset for a given language.

    Args:
        ds: The dataset containing translations.
        lang: The language key to extract sentences from.

    Yields:
        str: Sentences in the specified language.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """Retrieves or builds a tokenizer for a given language.

    Args:
        config: Configuration dictionary containing tokenizer file path.
        ds: The dataset to train the tokenizer on.
        lang: The language key for the tokenizer.

    Returns:
        Tokenizer: A trained tokenizer for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def download_dataset(config: dict):
    """
    Downloads and splits the dataset according to the configuration.

    Args:
        config (dict): Configuration dictionary. Relevant keys include:
            - 'dataset_path': Path or identifier for the dataset to load.
            - 'dataset_name': Name of the dataset configuration.
            - 'train_only_split' (bool): If True, splits the 'train' split into train/val. If False, loads separate 'train' and 'validation' splits.

    Returns:
        tuple:
            - train_ds_raw: Raw training dataset split.
            - val_ds_raw: Raw validation dataset split.
            - ds_raw: The combined or original dataset (for tokenizer building, etc).

    Notes:
        - If 'train_only_split' is True, the function splits the 'train' portion into train/val (90/10).
        - If False, loads both 'train' and 'validation' splits and combines them for tokenizer use.
    """

    if config['train_only_split']:
        ds_raw = load_dataset(config['dataset_path'], name=config['dataset_name'], split='train')
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size  # Ensures exact match
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
        return train_ds_raw, val_ds_raw, ds_raw
    else:
        train_ds_raw = load_dataset(config['dataset_path'], name=config['dataset_name'], split='train')
        val_ds_raw = load_dataset(config['dataset_path'], name=config['dataset_name'], split='validation')
        return train_ds_raw, val_ds_raw, train_ds_raw + val_ds_raw

def get_ds(config: dict):
    """Prepares training and validation datasets along with tokenizers.

    Args:
        config: Configuration dictionary containing language and dataset settings.

    Returns:
        tuple: A tuple containing:
            - train_dataloader: DataLoader for the training dataset.
            - val_dataloader: DataLoader for the validation dataset.
            - tokenizer_src: Tokenizer for the source language.
            - tokenizer_tgt: Tokenizer for the target language.
    """
    train_ds_raw, val_ds_raw, ds_raw = download_dataset(config)

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])


    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],  config['seq_len'])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],  config['seq_len'])
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['tgt_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Maximum lenght of source sentence: {max_len_src}')
    print(f'Maximum lenght of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config: dict, vocab_src_len, vocab_tgt_len):
    """Builds and returns a transformer model for bilingual translation.

    Args:
        config: Configuration dictionary containing model hyperparameters.
        vocab_src_len (int): Size of the source language vocabulary.
        vocab_tgt_len (int): Size of the target language vocabulary.

    Returns:
        nn.Module: A transformer model configured for the given task.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """
    Perform autoregressive greedy decoding to generate output sequences token-by-token.
    
    Args:
        model (Transformer): Initialized transformer model with encode/decode methods.
        source (torch.Tensor): Encoder input tensor of shape (1, src_seq_len).
        source_mask (torch.Tensor): Encoder attention mask of shape (1, 1, src_seq_len).
        tokenizer_tgt (Tokenizer): Target language tokenizer for special token lookup.
        max_len (int): Maximum allowed sequence length before forced termination.
        device (torch.device): Device (e.g., 'cuda' or 'cpu') for tensor operations.

    Returns:
        torch.Tensor: Generated sequence tensor of shape (1, output_seq_len) containing token IDs.

    Detailed Logic:
        1. Special Tokens Setup:
           - eos_idx: Stores the token ID for [EOS] (End-of-Sequence) from target tokenizer
           - sos_idx: Stores the token ID for [SOS] (Start-of-Sequence) from target tokenizer

        2. Encoder Forward Pass:
           - encoder_output = model.encode(source, source_mask)
             * Pre-computes encoder representations once (shape: (1, src_seq_len, d_model))
             * source_mask prevents attention to padding tokens

        3. Decoder Initialization:
           - decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
             * Creates tensor of shape (1,1) containing only [SOS] token
             * Ensures dtype/device compatibility with source input

        4. Autoregressive Generation Loop:
           while True:
             a. Length Check:
                - if decoder_input.size(1) == max_len: break
                  * Terminates if sequence reaches maximum allowed length

             b. Causal Mask Creation:
                - decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
                  * Creates triangular mask of shape (1, curr_len, curr_len)
                  * Example for length 3:
                    [[[1,0,0],
                      [1,1,0],
                      [1,1,1]]]

             c. Decoder Forward Pass:
                - out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
                  * Processes current sequence (shape: (1, curr_len, d_model))
                  * Uses encoder output and causal masking

             d. Next Token Prediction:
                - prob = model.project(out[:,-1])
                  * Projects last token's representation to vocabulary space
                  * Shape changes from (1, d_model) to (1, tgt_vocab_size)

                - _, next_word = torch.max(prob, dim=1)
                  * Selects token ID with highest probability (greedy selection)
                  * Ignores actual probability value (using underscore)

             e. Sequence Update:
                - decoder_input = torch.cat([decoder_input, new_token], dim=1)
                  * Appends predicted token to growing sequence
                  * new_token is shape (1,1) containing next_word

             f. Termination Check:
                - if next_word == eos_idx: break
                  * Stops generation if [EOS] token is predicted

    Note: This implementation uses greedy decoding (always selecting the highest-probability token).
    For alternatives like beam search, additional logic would be required.
    """
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ],
            dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(
    model: nn.Module,
    validation_ds: DataLoader,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: callable,
    num_examples: int = 2
) -> None:
    """Runs model validation and prints translation examples.
    
    Args:
        model: Transformer model to validate
        validation_ds: Validation DataLoader
        tokenizer_tgt: Target language tokenizer
        max_len: Maximum sequence length for decoding
        device: Device for tensor operations
        print_msg: Function to display messages (e.g., tqdm.write)
        num_examples: Number of examples to display
        
    Process:
        1. Sets model to eval mode (disables dropout)
        2. Iterates through validation batches:
            - Performs greedy decoding
            - Compares source/target/predicted texts
            - Prints results via print_msg
        3. Only evaluates specified number of examples for brevity
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask,tokenizer_tgt,max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f'SOURCE {source_text}')
            print_msg(f'TARGET {target_text}')
            print_msg(f'PREDICTED {model_out_text}')

            if count == num_examples:
                break        

def train_model(config: dict) -> None:
    """Trains the transformer model using the provided configuration.
    
    Args:
        config: Dictionary containing training parameters:
            - model_folder: Directory to save model weights
            - experiment_name: Tensorboard log directory
            - lr: Learning rate for optimizer
            - num_epochs: Total training epochs
            - preload: Optional epoch number to resume training from
            - Other model/dataset parameters (see get_config())
            
    Workflow:
        1. Initializes device (MPS/CPU), model, tokenizers, and data loaders
        2. Sets up Tensorboard logging and Adam optimizer
        3. Runs training loop with:
            - Forward/backward passes
            - Loss calculation (CrossEntropy with label smoothing)
            - Periodic validation
            - Model checkpointing
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok = True)

    train_dataloader, val_dataloader , tokenizer_src, tokenizer_tgt = get_ds(config) 

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimzer = torch.optim.Adam(model.parameters(),lr = config['lr'], eps = 1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimzer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1 ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing batch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask =  batch['encoder_mask'].to(device)
            decoder_mask =  batch['decoder_mask'].to(device)

            # Run the tensors trough the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimzer.step()
            optimzer.zero_grad()

            global_step +=1
  
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, batch_iterator.write)
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict':optimzer.state_dict(),
            'global_step':global_step
        },model_filename)

if __name__ == '__main__':
    model_config = get_config()
    train_model(model_config)
