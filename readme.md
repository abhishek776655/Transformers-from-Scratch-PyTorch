# 🚀 Transformer from Scratch: Pure PyTorch Powering Machine Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/transformer-from-scratch?style=social)](https://github.com/abhishek776655/Transformers-from-Scratch-PyTorch)

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
git clone https://github.com/abhishek776655/Transformers-from-Scratch-PyTorch
cd Transformers-from-Scratch-PyTorch

# Or with SSH (if you prefer)
git clone https://github.com/abhishek776655/Transformers-from-Scratch-PyTorch

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

### 3. Run Training

```bash
# Start training (logs auto-save to /runs)
python train.py

# Monitor in TensorBoard (open in browser)
tensorboard --logdir=runs --port=6006
```

---

## 📦 Code Structure

| File         | Purpose                                |
| ------------ | -------------------------------------- |
| `model.py`   | Core Transformer architecture          |
| `train.py`   | Training/validation pipeline           |
| `dataset.py` | Bilingual dataloader + masking         |
| `config.py`  | Hyperparameters (LR, batch size, etc.) |

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

3. **For Better Performance**:

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

4. **For Debugging**:

```python
# Add these in model.py to verify shapes
print(f"Encoder input: {x.shape}")          # Should be [batch, seq_len, d_model]
print(f"Attention weights: {attn.shape}")   # Should be [batch, heads, seq_len, seq_len]
```

---

## 🤝 Contribute

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

---

**⭐ Star this repo if it helped you master Transformers!**

> "Go from theory to practice—one attention head at a time!"
