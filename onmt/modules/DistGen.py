import torch
import torch.nn as nn
import math

class DistGen(nn.Module):
    
    def __init__(self):
        super(DistGen, self).__init__()
    
    def forward(self, vocab_ds, attns, p_gens, sources, decoder_batch_len):
        """
        vocab_ds: (decoder_batch_len*batch_size, decoder_voc_size)
        attns: (decoder_batch_len*batch_size, encoder_len)
        p_gens: (decoder_batch_len*batch_size, 1)
        sources: (encoder_len, batch_size)
        return: 
        """
        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = p_gens.expand_as(vocab_ds) * vocab_ds
        attn_dists = (1 - p_gens.expand_as(attns)) * attns
        
        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        
        # Project attn distribution to vocab distribution
        sources_rep = torch.t(sources).repeat(decoder_batch_len, 1) #(decoder_batch_len*batch_size, encoder_len)
        attn_dists_projected = torch.zeros(vocab_ds.size()).scatter_(1, sources_rep.data.cpu(), attn_dists.data.cpu())
        attn_dists_projected = torch.autograd.Variable(attn_dists_projected).cuda()
        final_dists = attn_dists_projected + vocab_dists
        return final_dists