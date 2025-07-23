import torch
import torch.nn as nn

class TransformerWithPreNorm(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_vocab_size, dropout=0.1):
        super(TransformerWithPreNorm, self).__init__()
        
        # This is the main transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True  # This enables pre-normalization
        )
        
        # Final linear layer
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, 
                memory_key_padding_mask=None):
        
        # The transformer returns the processed sequence
        output = self.transformer(src, tgt, 
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Pass the output of the transformer to the final linear layer
        output = self.fc_out(output)
        
        return output