# 🚀 Transformer from Scratch: Pure PyTorch Powering Machine Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/transformer-from-scratch?style=social)](https://github.com/abhishek776655/transformers-from-scratch-pytorch)

**Unlock the magic of self-attention!** This project is a hands-on implementation of the groundbreaking ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper, built entirely in **PyTorch**—no shortcuts, no high-level wrappers—just pure deep learning fundamentals.

---

## 🔍 What's Inside?

### 🧠 Pure PyTorch Implementation

Every component coded from scratch:

- **Multi-Head Attention** (with scaled dot-product)
- **Positional Encoding** (sine/cosine waves)
- **Encoder/Decoder Stacks** (6 layers each)
- **Label Smoothing** and **Masked Attention**

### 🌍 Bilingual Translation

- Train on custom language pairs (e.g., English ↔ Russian)
- Dynamic tokenization with **Hugging Face `tokenizers`**
- OPUS books dataset loader

### ⚡ Production-Grade Features

- **GPU/MPS acceleration** support
- **TensorBoard logging** (loss curves, attention heatmaps)
- **Model checkpointing** and **greedy decoding**

---

## 🛠️ Tech Stack

| Component                                                                       | Purpose                   |
| ------------------------------------------------------------------------------- | ------------------------- |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)            | Core model implementation |
| ![HuggingFace](https://img.shields.io/badge/Tokenizers-FF4C4C?logo=huggingface) | Fast subword tokenization |
| ![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow) | Training visualization    |

---

## 🎯 Perfect For

| Audience        | Benefit                                        |
| --------------- | ---------------------------------------------- |
| **Learners**    | Understand Transformers beyond abstract theory |
| **Engineers**   | Modular code for custom NLP tasks              |
| **Researchers** | Clean baseline for attention experiments       |

---

## 🔥 Quick Setup

### 1. Clone and Install

```bash
# Clone with HTTPS
git clone https://github.com/abhishek776655/transformers-from-scratch-pytorch
cd transformers-from-scratch-pytorch

# Or with SSH (if you prefer)
git clone https://github.com/abhishek776655/transformers-from-scratch-pytorch

# Install dependencies
pip install -r requirements.txt
```

### 2. Customize Your Config

Edit `config.py` to match your needs:

```python
config = {
    # Data parameters
    'src_lang': 'en',           # Source language (e.g., 'de' for German)
    'tgt_lang': 'ru',           # Target language (e.g., 'fr' for French)
    'batch_size': 32,           # Increase for better GPU utilization

    # Dataset configuration
    'dataset_path': 'opus_books',     # Hugging Face dataset identifier
    'dataset_name': 'en-ru',          # Dataset configuration (language pair)
    'train_only_split': True,         # If True, splits 'train' into train/val; if False, loads both splits

    # Model architecture
    'd_model': 512,             # Embedding dimension
    'n_heads': 8,               # Number of attention heads
    'num_layers': 6,            # Encoder/decoder layers
    'dropout': 0.1,             # Regularization

    # Training
    'lr': 0.0001,               # Learning rate
    'num_epochs': 20,           # Total epochs
    'preload': 'latest'         # Set to epoch number to resume (e.g., '10')
}
```

---

### 📚 Dataset Configuration

- **`dataset_path`**: (str) Hugging Face dataset identifier (e.g., `'opus_books'`, `'wmt14'`).
- **`dataset_name`**: (str) Dataset configuration or language pair (e.g., `'en-ru'`, `'en-de'`).
- **`train_only_split`**: (bool) If `True`, splits the `'train'` split into train/validation (90/10). If `False`, loads both `'train'` and `'validation'` splits directly (recommended for datasets that provide both).

You can easily switch datasets or add new language pairs by changing these fields in your config.

### 3. Run Training

```bash
# Start training with advanced logging (recommended for tracked experiments)
python train_with_logging.py

# Or use the basic training pipeline
python train.py

# Monitor in TensorBoard (open in browser)
tensorboard --logdir=runs --port=6006
```

- `train_with_logging.py` is the recommended entrypoint for experiments you want to track and analyze. It logs detailed metrics to both TensorBoard and Weights & Biases (wandb).
- `train.py` is a simpler script for quick runs or debugging.

---

## 📊 Training with Advanced Logging

For enhanced experiment tracking and debugging, use `train_with_logging.py`. This script extends the standard training pipeline by adding detailed logging to both **TensorBoard** and **Weights & Biases (wandb)**, making it easier to monitor, visualize, and compare your model’s performance.

### Key Logging Features

- **Training and Validation Loss** (per batch and per epoch)
- **Learning Rate** (per batch)
- **Batch and Epoch Timing** (profiling training speed)
- **Gradient and Weight Norms** (per layer and totals)
- **Histograms** of gradients and weights (periodically; see `HISTOGRAM_LOG_INTERVAL` in code to adjust frequency)
- **Validation Metrics:** BLEU score, token-level accuracy
- **System Resource Usage:** (e.g., memory monitoring via `psutil`)
- **Seamless integration with wandb:** All metrics and histograms are logged to your wandb dashboard for experiment tracking.

> **Note:** To change how often histograms are logged, edit the `HISTOGRAM_LOG_INTERVAL` constant in `train_with_logging.py` (default: 100 steps).

### Usage

```bash
python train_with_logging.py
```

- All logs are saved to the `runs/` directory for TensorBoard.
- To launch TensorBoard and monitor training in your browser:

```bash
tensorboard --logdir=runs --port=6006
```

- For wandb, ensure you have an account and are logged in (`wandb login`). Metrics will be available in your wandb project dashboard.

### Example wandb Setup

```bash
pip install wandb
wandb login
python train_with_logging.py
```

### Sample Log Outputs

- **TensorBoard:** Visualize loss curves, learning rate schedules, gradient/weight histograms, and more.
- **wandb:** Interactive dashboards for comparing runs, monitoring system usage, and sharing results.

**Tip:** Use this script when you want maximum visibility into your model’s training dynamics, or when running experiments you want to track and compare over time.

---

---

## 📦 Code Structure

| File/Folder             | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| `model.py`              | Core Transformer architecture                       |
| `train.py`              | Training/validation pipeline                        |
| `train_with_logging.py` | Training with advanced logging (TensorBoard, wandb) |
| `dataset.py`            | Bilingual dataloader + masking                      |
| `config.py`             | Hyperparameters (LR, batch size, etc.)              |
| `inference.ipynb`       | Notebook for running inference/translation          |

---

## 💡 Why This Stands Out

✅ **Paper-Faithful Implementation**

- Direct mapping to original equations (e.g., `softmax(QKᵀ/√dₖ)V`)
- Shape assertions at every layer

✅ **Real-World Ready**

- Handles padding, batching, and memory optimization
- GPU/MPS support for accelerated training

✅ **Educational Focus**

- Detailed docstrings linking to paper sections
- TensorBoard attention visualization

---

## 🌟 Pro Tips

1. **Customize Languages**:

```python
# config.py
config = {
    'src_lang': 'de',  # German
    'tgt_lang': 'it'   # Italian
}
```

2. **Debug Attention**:

```python
# model.py
print(f"Attention weights shape: {attention_weights.shape}")
```

3. **For Better Performance (Mixed Precision)**:

```python
# Enable mixed precision training (if GPU available)
scaler = torch.cuda.amp.GradScaler()

with torch.amp.autocast(device_type='cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- Mixed precision is supported in `train_with_logging.py` for faster training and lower memory usage on modern GPUs. Enable it in your config or script as shown above.

4. **For Debugging**:

```python
# Add these in model.py to verify shapes
print(f"Encoder input: {x.shape}")          # Should be [batch, seq_len, d_model]
print(f"Attention weights: {attn.shape}")   # Should be [batch, heads, seq_len, seq_len]
```

---

## 🛠️ Troubleshooting & Logging Tips

- **wandb login issues:** If wandb metrics are not logging, run `wandb login` in your terminal and ensure your API key is set.
- **CUDA out-of-memory:** Lower the batch size in `config.py` or use mixed precision.
- **TensorBoard not showing logs:** Double-check the `runs/` directory and ensure the correct path is set in TensorBoard.
- **Custom logging:** Adjust `HISTOGRAM_LOG_INTERVAL` or add more `writer.add_scalar`/`wandb.log` calls in `train_with_logging.py` as needed.
- **Model checkpoints:** Checkpoints are saved after each epoch in the folder specified by your config.
- **Config changes:** If you add new config options, document them in `config.py` and this README.

---

## 🤝 Contribute

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**⭐ Star this repo if it helped you master Transformers!**

> "Go from theory to practice—one attention head at a time!"
