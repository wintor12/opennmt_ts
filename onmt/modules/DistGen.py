import torch
import torch.nn as nn
import math

class DistGen(nn.Module):
    
    def __init__(self):
        super(DistGen, self).__init__()
    
    def forward(self, vocab_ds, attns, p_gens):
        """
        vocab_ds: (decoder_batch_len, batch_size, decoder_voc_size)
        attns: (decoder_batch_len, batch_size, encoder_len)
        p_gens: (decoder_batch_len, batch_size, 1)
        return: 
        """
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = p_gens.expand_as(vocab_ds) * vocab_ds
        attn_dists = (1 - p_gens.expand_as(attns)) * attns
        
        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        
        
        final_dists = attn_dists
        return final_dists