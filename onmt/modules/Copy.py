"""
Copy mechanism

"""

import torch
import torch.nn as nn
import math

class Copy(nn.Module):
    def __init__(self, dim, emb_dim, layers):
        super(Copy, self).__init__()
        self.linear = nn.Linear(dim + 2*layers*dim + emb_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, attn, emb_t, hidden, context):
        """
        attn: batch x sourceL
        emb_t: batch x emb_dim
        hidden: (h:layers x batch x dim, c:layers x batch x dim)
        context: batch x sourceL x dim
        """
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h = hidden[0].transpose(0,1).contiguous() #batch x layers x dim
        h = h.view(h.size(0),-1) #batch x (layers*dim)
        c = hidden[1].transpose(0,1).contiguous()
        c = h.view(c.size(0),-1)
        linear = self.linear(torch.cat((weightedContext, h, c, emb_t),1))
        p_gen = self.sigmoid(linear)
        return p_gen
