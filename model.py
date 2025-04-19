"""Implementation of the Transformer model from "Attention Is All You Need" (Vaswani et al., 2017).

This module provides a complete implementation of the Transformer architecture, including:
- Input embeddings and positional encoding
- Multi-head attention mechanisms
- Encoder and decoder stacks with residual connections and layer normalization
- Feed-forward networks
- A projection layer for output vocabulary

Key Classes:
- Transformer: The complete model combining encoder, decoder, and projection layers.
- Encoder/Decoder: Stacks of encoder/decoder blocks.
- MultiHeadAttentionBlock: Implements scaled dot-product attention.
- FeedForwardBlock: Position-wise feed-forward network.
- InputEmbeddings: Token embeddings with scaling.
- PositionalEncoding: Adds positional information to embeddings.

Usage:
    from model import build_transformer
    transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)
"""

import math

import torch
from torch import nn

class InputEmbeddings(nn.Module):
    """Converts input tokens to embeddings with learned weights.

    Args:
        d_model (int): Dimension of the model embeddings.
        vocab_size (int): Size of the vocabulary.

    Attributes:
        d_model (int): Dimension of the model embeddings.
        vocab_size (int): Size of the vocabulary.
        embedding (nn.Embedding): Embedding layer.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """Compute input embeddings and scale them by sqrt(d_model).

        Args:
            x (torch.Tensor): Input tensor of token indices, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Scaled embeddings, shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    """Adds positional information to input embeddings.

    Args:
        d_model (int): Dimension of the model embeddings.
        seq_len (int): Maximum sequence length.
        drop_out (float): Dropout probability.

    Attributes:
        d_model (int): Dimension of the model embeddings.
        seq_len (int): Maximum sequence length.
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Positional encoding buffer.
    """
    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(drop_out)

        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(1000.0) / d_model))
        # Apply sin to even positions
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_len , d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings, shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding, shape (batch_size, seq_len, d_model).
        """
        x = x + (self.pe[:, :x.shape[1], :]).detach()
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """Applies layer normalization to the input tensor.

    Args:
        eps (float): Small value to avoid division by zero. Default: 1e-6.

    Attributes:
        eps (float): Small value to avoid division by zero.
        alpha (nn.Parameter): Learnable scaling parameter.
        bias (nn.Parameter): Learnable bias parameter.
    """
    def __init__(self, eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Normalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor, shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor, same shape as input.
        """
        mean = x.mean(dim = -1 , keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    """Implements the feed-forward network component of a Transformer block.

    Args:
        d_model (int): Dimension of the model embeddings.
        d_ff (int): Dimension of the feed-forward hidden layer.
        drop_out (float): Dropout probability.

    Attributes:
        linear_1 (nn.Linear): First linear transformation.
        dropout (nn.Dropout): Dropout layer.
        linear_2 (nn.Linear): Second linear transformation.
    """
    def __init__(self, d_model: int, d_ff: int, drop_out: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(drop_out)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        """Apply the feed-forward network to the input.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor, same shape as input.
        """
        # (Batch , seq_len , d_model) --> (Batch , seq_len , d_ff) --> (Batch , seq_len , d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """Implements multi-head self-attention mechanism.

    Args:
        d_model (int): Dimension of the model embeddings.
        h (int): Number of attention heads.
        drop_out (float): Dropout probability.

    Attributes:
        d_model (int): Dimension of the model embeddings.
        h (int): Number of attention heads.
        d_k (int): Dimension of each attention head.
        w_q (nn.Linear): Linear layer for queries.
        w_k (nn.Linear): Linear layer for keys.
        w_v (nn.Linear): Linear layer for values.
        w_o (nn.Linear): Linear layer for output.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, d_model: int, h: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(drop_out)

        self.attention_scores = None
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor, shape (batch_size, h, seq_len, d_k).
            key (torch.Tensor): Key tensor, shape (batch_size, h, seq_len, d_k).
            value (torch.Tensor): Value tensor, shape (batch_size, h, seq_len, d_k).
            mask (torch.Tensor): Mask tensor, shape (batch_size, seq_len, seq_len).
            dropout (nn.Dropout): Dropout layer.

        Returns:
            tuple: (output, attention_scores)
                - output: Tensor after attention, shape (batch_size, h, seq_len, d_k).
                - attention_scores: Attention weights, shape (batch_size, h, seq_len, seq_len).
        """
        d_k = query.shape[-1]

        # (Batch , h, seq_len, d_k) --> (Batch , h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # (Batch, h, seq_len, d_k)

    def forward(self, q, k, v, mask):
        """Apply multi-head attention to the input.

        Args:
            q (torch.Tensor): Query tensor, shape (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor, shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor, shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        query = self.w_q(q) # (Batch,seq_len,d_model) -> (Batch,seq_len,d_model)
        key = self.w_k(k) # (Batch,seq_len,d_model) -> (Batch,seq_len,d_model)
        value = self.w_v(v) # (Batch,seq_len,d_model) -> (Batch,seq_len,d_model)

        # (Batch,seq_len,d_model) - > (Batch,seq_len, h, d_k) --> (Batch, h, seq_len ,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    """Implements a residual connection with layer normalization and dropout.

    Args:
        dropout (float): Dropout probability.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        norm (LayerNormalization): Layer normalization module.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """Applies the residual connection: `x + dropout(sublayer(norm(x)))`.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (callable): A function or module to apply to the normalized input.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    """Implements a single block of the Transformer encoder.

    Each encoder block consists of:
    - A multi-head self-attention mechanism with residual connection.
    - A feed-forward network with residual connection.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The self-attention module.
        feed_forward_block (FeedForwardBlock): The feed-forward network module.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention module.
        feed_forward_block (FeedForwardBlock): The feed-forward network module.
        residual_connections (nn.ModuleList): List of residual connections for the sublayers.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """Processes the input through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor for padding or future tokens, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
        
class Encoder(nn.Module):
    """Implements the encoder component of the Transformer model.

    The encoder consists of a stack of identical layers, each containing:
    - A self-attention mechanism with residual connection and layer normalization.
    - A feed-forward network with residual connection and layer normalization.
    - Layer normalization is applied to the final output of the encoder stack.

    Attributes:
        layers (nn.ModuleList): A list of EncoderBlock layers.
        norm (LayerNormalization): Layer normalization applied to the final output.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        """Processes the input through the encoder stack.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor for padding or future tokens, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """Implements a single block of the Transformer decoder.

    Each decoder block consists of:
    - A masked multi-head self-attention mechanism with residual connection.
    - A multi-head cross-attention mechanism with residual connection.
    - A feed-forward network with residual connection.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The masked self-attention module.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention module.
        feed_forward_block (FeedForwardBlock): The feed-forward network module.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The masked self-attention module.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention module.
        feed_forward_block (FeedForwardBlock): The feed-forward network module.
        residual_connections (nn.ModuleList): List of residual connections for the sublayers.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Processes the input through the decoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor for the encoder output, shape (batch_size, seq_len, seq_len).
            tgt_mask (torch.Tensor): Mask tensor for the decoder input, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """Implements the decoder component of the Transformer model.

    The decoder consists of a stack of identical layers, each containing:
    - A masked multi-head self-attention mechanism with residual connection.
    - A multi-head cross-attention mechanism with residual connection.
    - A feed-forward network with residual connection.
    - Layer normalization is applied to the final output of the decoder stack.

    Args:
        layers (nn.ModuleList): A list of DecoderBlock layers.

    Attributes:
        layers (nn.ModuleList): A list of DecoderBlock layers.
        norm (LayerNormalization): Layer normalization applied to the final output.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Processes the input through the decoder stack.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor for the encoder output, shape (batch_size, seq_len, seq_len).
            tgt_mask (torch.Tensor): Mask tensor for the decoder input, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """Projects the decoder output to the target vocabulary space.

    Args:
        d_model (int): Dimension of the model embeddings.
        vocab_size (int): Size of the target vocabulary.

    Attributes:
        proj (nn.Linear): Linear projection layer.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Projects the input tensor to the target vocabulary space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Log-softmax output tensor of shape (batch_size, seq_len, vocab_size).
        """

        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    """Implements the complete Transformer model.

    The Transformer consists of:
    - An encoder for processing the source sequence.
    - A decoder for generating the target sequence.
    - Embedding and positional encoding layers for both source and target.
    - A projection layer for mapping decoder output to the target vocabulary.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embedding (InputEmbeddings): Source embedding layer.
        tgt_embedding (InputEmbeddings): Target embedding layer.
        src_pos (PositionalEncoding): Source positional encoding layer.
        tgt_pos (PositionalEncoding): Target positional encoding layer.
        projection_layer (ProjectionLayer): Projection layer for the decoder output.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embedding (InputEmbeddings): Source embedding layer.
        tgt_embedding (InputEmbeddings): Target embedding layer.
        src_pos (PositionalEncoding): Source positional encoding layer.
        tgt_pos (PositionalEncoding): Target positional encoding layer.
        projection_layer (ProjectionLayer): Projection layer for the decoder output.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """Encodes the source sequence.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, seq_len).
            src_mask (torch.Tensor): Mask tensor for the source sequence, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, d_model).
        """
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """Decodes the target sequence using the encoder output.

        Args:
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Mask tensor for the encoder output, shape (batch_size, seq_len, seq_len).
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, seq_len).
            tgt_mask (torch.Tensor): Mask tensor for the target sequence, shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, d_model).
        """
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """Projects the decoder output to the target vocabulary space.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Log-softmax output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.projection_layer(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Processes the input through the complete Transformer model.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len).
            src_mask (torch.Tensor): Mask tensor for source sequence.
            tgt_mask (torch.Tensor): Mask tensor for target sequence.

        Returns:
            torch.Tensor: Projected output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)
    
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Constructs and initializes a Transformer model with the given configuration.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of the source sequence.
        tgt_seq_len (int): Maximum length of the target sequence.
        d_model (int, optional): Dimension of the model embeddings. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        d_ff (int, optional): Dimension of the feed-forward hidden layer. Defaults to 2048.

    Returns:
        Transformer: A fully initialized Transformer model with Xavier-uniform initialized parameters.

    Notes:
        - The model includes:
            - Embedding layers for source and target.
            - Positional encoding layers for source and target.
            - Encoder and decoder stacks with `N` layers each.
            - A projection layer for the target vocabulary.
        - All linear layers with dimension > 1 are initialized using Xavier-uniform initialization.
    """
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


