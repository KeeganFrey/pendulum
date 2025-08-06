import math
import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    """
    This class is a PyTorch module that will take a tensor of 3D coordinates 
    (patch_x, patch_y, temporal_position) and embed them.
    """
    def __init__(self, channels=3, embedding_dim=1024):
        """
        Initializes the PositionalEmbedding module.

        Args:
            channels (int): The number of input channels for the coordinate vector (e.g., 3 for x, y, t).
            embedding_dim (int): The dimension of the output embedding.
        """
        super().__init__()
        self.channels = channels
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(channels, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, positional_vectors):
        """
        Performs the forward pass of the positional embedding.

        Args:
            positional_vectors (torch.Tensor): A tensor of shape (num_patches, channels) 
                                               containing the coordinates for each patch.

        Returns:
            torch.Tensor: The resulting positional embeddings of shape (num_patches, embedding_dim).
        """
        # Input shape: (num_patches, 3)
        # Output shape: (num_patches, 1024)

        embedding = self.linear(positional_vectors.float())
        embedding = self.activation(embedding)
        return embedding
    
class PatchEmbedding(nn.Module):
    """
    This class is a Pytorch module that will take a 1D image, and perform a patch embedding on it.
    """
    def __init__(self, patch_size=16, channels=3, embedding_dim=1024):
        """
        Initializes the PatchEmbedding module.

        Args:
            patch_size (int): The size of each square patch.
            channels (int): The number of channels in the input image.
            embedding_dim (int): The dimension of the output embedding for each patch.
        """
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.embedding_dim = embedding_dim
        
        # The Unfold layer extracts sliding local blocks from a batched input tensor.
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        
        # The flattened dimension of each patch.
        flattened_dim = patch_size * patch_size * channels
        
        # A linear layer to project the flattened patches to the embedding dimension.
        self.linear = nn.Linear(flattened_dim, embedding_dim)
        
        # The ReLU activation function.
        self.activation = nn.ReLU()

    def forward(self, image, height, width):
        """
        Performs the forward pass of the patch embedding.

        Args:
            image (torch.Tensor): The input image as a 1D tensor.
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            torch.Tensor: The resulting patch embeddings.
        """
        # 1. Reshape the 1D image tensor to a 3D tensor (channels, height, width)
        #    and add a batch dimension.
        image_tensor = image.view(1, self.channels, height, width)

        # 2. Extract patches from the image.
        #    The output of unfold will have the shape (batch_size, patch_size*patch_size*channels, num_patches)
        patches = self.unfold(image_tensor)

        # 3. Transpose the patches to have the shape (batch_size, num_patches, patch_size*patch_size*channels)
        #    so that we can apply the linear layer to each patch.
        patches = patches.transpose(1, 2)

        # 4. Apply the linear transformation to the flattened patches.
        embedding = self.linear(patches)

        # 5. Apply the ReLU activation function.
        embedding = self.activation(embedding)
        
        # 6. The output `embedding` is a tensor of shape (1, num_patches, embedding_dim).
        #    We squeeze the batch dimension to get (num_patches, embedding_dim).
        return embedding.squeeze(0)

"""
 Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
 """
class CausalSelfAttention(nn.Module):
 
    def __init__(
        self,
        d,
        H,
        T,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0
 
        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=bias)
 
        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)
 
        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d
 
        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T))
                                     .view(1, 1, T, T))
 
    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality
 
        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.d, dim=2) # [B, T, d]
 
        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)
 
        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
 
        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]
 
        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FFNN(nn.Module):
 
    def __init__(
        self,
        d,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.c_fc    = nn.Linear(d, 4 * d, bias=bias)
        self.gelu    = nn.ReLU()
        self.c_proj  = nn.Linear(4 * d, d, bias=bias)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        x = self.c_fc(x) # [B, T, 4*d]
        x = self.gelu(x)
        x = self.c_proj(x) # [B, T, d]
        x = self.dropout(x)
        return x

class L2Norm(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class Block(nn.Module):
    def __init__(
        self,
        d,
        H,
        T,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.ln_1 = L2Norm(d)
        self.attn = CausalSelfAttention(d, H, T, bias, dropout)
        self.ln_2 = L2Norm(d)
        self.ffnn = FFNN(d, bias, dropout)
 
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffnn(self.ln_2(x))
        return x
     
class TransformerModel(nn.Module):
    def __init__(self, num_token_options, num_heads, embd_dim, t, nlayers = 5):
       super().__init__()
       self.model_type = 'Transformer'
       self.pos_encoder = PositionalEmbedding(channels=3, embedding_dim=embd_dim)
       self.encoder = PatchEmbedding(patch_size=16, channels=3, embedding_dim=embd_dim)
       self.transformer = nn.Sequential(*[Block(embd_dim, num_heads, t) for _ in range(nlayers)])
       self.d_model = embd_dim
       self.T = t
       self.decoder = nn.Linear(embd_dim, num_token_options) 

    def forward(self, images, height, width, positional_infos):
        """
        Args:
            images (list of torch.Tensor): The input image as a 1D tensor.
            height (int): The height of the image.
            width (int): The width of the image.
            positional_infos (list of torch.Tensor): A list of tensors of shape (num_patches, 3) 
                                             containing the (x, y, t) coordinates for each patch.
        Returns:
            torch.Tensor: The output logits from the model.
        """
        tokens = 0
        t_seq = torch.zeros((0, self.d_model), dtype=float)
        for i in range(images.size()):
            patch_embeddings = self.encoder(images[i], height, width)

            positional_embeddings = self.pos_encoder(positional_infos[i])

            x = (patch_embeddings + positional_embeddings)
            t_seq = torch.cat([t_seq, x], dim=0) 
            tokens += positional_infos[i].shape[0] # assumes that the positional infos holds tensors rather than lists

        t_seq = torch.cat([t_seq, torch.ones((1, self.d_model), dtype=float)], dim=0)
        t_seq = torch.cat([t_seq, torch.zeros((self.T - (tokens+1), self.d_model), dtype=float)], dim=0)

        # 4. Pass through the transformer blocks
        transformer_output = self.transformer(t_seq.unsqueeze(0))

        # 5. Decode the output of the transformer
        # We squeeze the batch dimension before passing to the decoder
        output_logits = self.decoder(transformer_output.squeeze(0))

        return output_logits