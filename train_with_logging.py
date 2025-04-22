"""Module for training a transformer model for bilingual translation tasks.

This module includes functions for tokenizer creation, dataset preparation, and data loading.
"""
# Standard Library
import time
from pathlib import Path
from torch.utils.data import random_split, DataLoader

# Third-Party Libraries
import torch
from torch import nn
import torch.cuda as cuda

from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import wandb
import psutil  # For memory monitoring
from sacrebleu.metrics import BLEU

# Local Modules
from config import get_config, get_weights_file_path
from dataset import BillingualDataset, causal_mask
from model import build_transformer

# Add near the top with other constants
HISTOGRAM_LOG_INTERVAL = 100 # Log histograms every 100 steps (match other metrics)

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
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    # build tokenizer   
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])  
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])
    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size  # Ensures exact match
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

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

def greedy_decode(
    model: nn.Module,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    log_metrics: bool = False,
    loss_fn=None,
    label=None
) -> tuple[torch.Tensor, dict]:
    """
    Perform autoregressive greedy decoding to generate output sequences token-by-token.
    
    Args:
        model (Transformer): Initialized transformer model with encode/decode methods.
        source (torch.Tensor): Encoder input tensor of shape (1, src_seq_len).
        source_mask (torch.Tensor): Encoder attention mask of shape (1, 1, src_seq_len).
        tokenizer_tgt (Tokenizer): Target language tokenizer for special token lookup.
        max_len (int): Maximum allowed sequence length before forced termination.
        device (torch.device): Device (e.g., 'cuda' or 'cpu') for tensor operations.
        log_metrics: bool = False  # New flag for validation logging
        loss_fn: callable = None  # Add loss function parameter
        label: torch.Tensor = None  # Add ground truth labels

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
    
    metrics = {
        "predicted_length": 0,
        "correct_tokens": 0,
        "total_tokens": 0,
        "loss": 0.0  # Add loss tracking
    }

    # Store decoder outputs for loss calculation
    decoder_outputs = []
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        # --- Validation Logging ---
        if log_metrics:
            metrics["predicted_length"] += 1
            # Compare predicted token with target (requires ground truth)
            # (Note: For full token accuracy, modify to pass target tokens)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break

        # Store decoder output for loss calculation
        if log_metrics and loss_fn is not None:
            decoder_outputs.append(model.project(out[:, -1:]))  # Store last output
    # Calculate loss if requested
    if log_metrics and loss_fn is not None and label is not None:
        # 1. Concatenate logits (shape: [1, decoded_seq_len, vocab_size])
        all_outputs = torch.cat(decoder_outputs, dim=1)
        # 2. Pad logits to match label's seq_len (if shorter)
        if all_outputs.size(1) < label.size(1):
            padding = torch.zeros(
                (1, label.size(1) - all_outputs.size(1), all_outputs.size(2)),
                device=all_outputs.device,
                dtype=all_outputs.dtype
            )
            all_outputs = torch.cat([all_outputs, padding], dim=1) # Shape: [1, seq_len, vocab_size]
        
        # 3. Compute loss (flatten logits and labels)
        metrics["loss"] = loss_fn(
            all_outputs.view(-1, tokenizer_tgt.get_vocab_size()),  # [seq_len, vocab_size]
            label.view(-1)  # [seq_len]
        ).item()
    
    return decoder_input.squeeze(0), metrics

def run_validation(
    model: nn.Module,
    validation_ds: DataLoader,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: callable,
    epoch,
    num_examples: int = 2,
    writer=None,
    global_step: int = 0
) -> dict:
    """
    Runs validation on the model, computes metrics, and logs results.

    Args:
        model (nn.Module): Transformer model to validate.
        validation_ds (DataLoader): Validation dataset loader.
        tokenizer_tgt (Tokenizer): Target language tokenizer.
        max_len (int): Maximum sequence length for decoding.
        device (torch.device): Device (e.g., 'cuda', 'mps', 'cpu').
        print_msg (callable): Function to print messages (e.g., tqdm.write).
        num_examples (int, optional): Number of examples to display. Defaults to 2.
        writer (SummaryWriter, optional): TensorBoard writer. Defaults to None.
        global_step (int, optional): Global step for logging. Defaults to 0.

    Returns:
        dict: Validation metrics including:
            - bleu_score (float): BLEU score (if implemented).
            - token_accuracy (float): Token-level accuracy (if implemented).
            - avg_pred_length (float): Average predicted sequence length.
            - examples (list): Sample translations for inspection.

    Workflow:
        1. Sets model to eval mode.
        2. Iterates through validation batches:
            - Computes metrics (BLEU, accuracy, length) via greedy_decode.
            - Logs metrics to TensorBoard/wandb if writers are provided.
            - Prints example translations.
        3. Returns aggregated metrics.

    Example:
        >>> val_metrics = run_validation(
                model, val_loader, tokenizer, max_len=50, device=device,
                print_msg=tqdm.write, writer=writer, global_step=step
            )
        >>> print(val_metrics["bleu_score"])
    """
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    # Initialize metrics
    total_correct_tokens = 0
    total_tokens = 0
    bleu = BLEU()  # Make sure to import: `from sacrebleu.metrics import BLEU`

    # Existing metrics (unchanged)
    val_metrics = {
        "bleu_score": 0.0,
        "token_accuracy": 0.0,
        "avg_pred_length": 0.0,
        "examples": [],
        "val_loss": 0.0  # Add validation loss tracking
    }

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Get model output and metrics
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out, metrics = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_tgt, max_len, device,
                log_metrics=True,
                loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1),
                label=batch['label'].to(device)
            )
            val_metrics["avg_pred_length"] += metrics["predicted_length"]
            # (Add BLEU/accuracy calculations here)
            
            # Example display (unchanged)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # --- Token-Level Accuracy Calculation ---
            # Convert target text to token IDs
            target_ids = tokenizer_tgt.encode(target_text).ids
            predicted_ids = model_out.detach().cpu().tolist()  # Convert tensor to list

            # Pad or truncate to match lengths
            min_len = min(len(target_ids), len(predicted_ids))
            target_ids = target_ids[:min_len]
            predicted_ids = predicted_ids[:min_len]

            # Count correct tokens
            correct = sum(1 for t, p in zip(target_ids, predicted_ids) if t == p)
            total_correct_tokens += correct
            total_tokens += min_len

            # --- Example Display ---
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            with open("validation_logs.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Epoch: {epoch}\n")
                log_file.write(f"Source: {source_text}\n")
                log_file.write(f"Target: {target_text}\n")
                log_file.write(f"Predicted: {model_out_text}\n")
                log_file.write("-" * console_width + "\n")

            if count == num_examples:
                break

    # --- Finalize Metrics ---
    # BLEU Score
    val_metrics["bleu_score"] = bleu.corpus_score(predicted, [expected]).score

    # Token Accuracy
    val_metrics["token_accuracy"] = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0

    # Avg Pred Length 
    val_metrics["avg_pred_length"] /= count
    val_metrics["examples"] = list(zip(source_texts, expected, predicted))

    # Add validation loss
    val_metrics["val_loss"] = metrics["loss"]

    # --- Logging 
    if writer:
        writer.add_scalar('val_bleu', val_metrics["bleu_score"], global_step)
        writer.add_scalar('val_token_accuracy', val_metrics["token_accuracy"], global_step)
        writer.add_scalar('val_avg_pred_length', val_metrics["avg_pred_length"], global_step)
        writer.add_scalar('val_loss', val_metrics["val_loss"], global_step)

    return val_metrics

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
    wandb.init(
        project="transformer-translation",
        config=config,
        name=config['experiment_name']
    )
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

    LOG_INTERVAL = 100  # Same as other metrics

    for epoch in range(initial_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing batch {epoch:02d}")
        epoch_train_loss = 0.0  # Track epoch training loss
        num_batches = 0
        for batch in batch_iterator:
            model.train()
            batch_start_time = time.time()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask =  batch['encoder_mask'].to(device)
            decoder_mask =  batch['decoder_mask'].to(device)

            # Run the tensors trough the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            epoch_train_loss += loss.item()  # Accumulate epoch loss
            num_batches += 1
            loss.backward()

            ## Logging 

            # 1. Batch time (original)
            batch_time = (time.time() - batch_start_time) * 1000  # milliseconds

            layer_metrics = {}

            # 2. Memory usage (GPU/CPU)
            if device.type == 'cuda':
                memory_used = cuda.memory_allocated(device) / (1024 ** 2)  # MB
                memory_reserved = cuda.memory_reserved(device) / (1024 ** 2)
                layer_metrics.update({
                    "gpu_memory_used_mb": memory_used,
                    "gpu_memory_reserved_mb": memory_reserved
                })
            layer_metrics["cpu_memory_used_mb"] = psutil.Process().memory_info().rss / (1024 ** 2)

            # 3. Batch throughput (examples/sec)
            batch_size = encoder_input.size(0)
            layer_metrics["batch_throughput"] = batch_size / (batch_time / 1000)  # Avoid division by zero
            
            # 4. Logging Gradients
            if global_step % LOG_INTERVAL == 0:
                # --- Weight/Gradient Logging (Less Frequent) ---
                with torch.no_grad():
                    # 1. Log gradient norms (per-layer and total)
                    grad_norms = {name: param.grad.norm(2).item() 
                                 for name, param in model.named_parameters() 
                                 if param.grad is not None}
                    total_grad_norm = sum(grad_norms.values())
                    
                    # 2. Log weight norms (per-layer and total)
                    weight_norms = {name: param.norm(2).item() 
                                   for name, param in model.named_parameters()}
                    total_weight_norm = sum(weight_norms.values())
                    
                    # Log to TensorBoard
                    writer.add_scalar('total_gradient_norm', total_grad_norm, global_step)
                    writer.add_scalar('total_weight_norm', total_weight_norm, global_step)
                    
                    # Log to wandb (per-layer and totals)
                    wandb.log({
                        **{f"grad_norm/{name}": val for name, val in grad_norms.items()},
                        **{f"weight_norm/{name}": val for name, val in weight_norms.items()},
                        "total_gradient_norm": total_grad_norm,
                        "total_weight_norm": total_weight_norm,
                    }, step=global_step)

            # --- Always Log These (Frequent) ---
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('learning_rate', optimzer.param_groups[0]['lr'], global_step)
            writer.add_scalar('batch_time_ms', batch_time, global_step)
            for metric_name, value in layer_metrics.items():
                writer.add_scalar(metric_name, value, global_step)

            # Log to wandb
            wandb_logs = {
                "train_loss": loss.item(),
                "learning_rate": optimzer.param_groups[0]['lr'],
                "batch_time_ms": batch_time,
                **layer_metrics
            }

            if global_step % HISTOGRAM_LOG_INTERVAL == 0:
                # Only log histograms periodically
                wandb_logs.update({
                    **{f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()) 
                       for name, param in model.named_parameters() if param.grad is not None},
                    **{f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy()) 
                       for name, param in model.named_parameters()}
                })

            wandb.log(wandb_logs, step=global_step)

            optimzer.step()
            optimzer.zero_grad()

            global_step += 1

        # Epoch-level metrics
        epoch_time = time.time() - epoch_start_time
        writer.add_scalar('epoch_time_sec', epoch_time, epoch)
        wandb.log({"epoch_time_sec": epoch_time, "epoch": epoch})
        avg_epoch_train_loss = epoch_train_loss / num_batches
        
        # Log epoch training loss
        writer.add_scalar('train_epoch_loss', avg_epoch_train_loss, epoch)
        wandb.log({
            "train_epoch_loss": avg_epoch_train_loss,
            "epoch": epoch
        })
        # Run validation with logging
        val_metrics = run_validation(
            model, val_dataloader, tokenizer_tgt,
            config['seq_len'], device, batch_iterator.write,
            epoch,
            num_examples=10,
            writer=writer, global_step=global_step
        )

        # Log to wandb
        wandb.log({
            "val_loss": val_metrics["val_loss"],
            "val_bleu": val_metrics["bleu_score"],
            "val_token_accuracy": val_metrics["token_accuracy"],
            "epoch": epoch
        })

        # Save model checkpoint
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
