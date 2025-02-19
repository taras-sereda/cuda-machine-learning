import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    n_batch = 8
    n_time = 10
    n_hidden = 32
    n_head = 4
    q = torch.rand(n_batch, n_time, n_hidden)
    k = torch.rand(n_batch, n_time, n_hidden)
    v = torch.rand(n_batch, n_time, n_hidden)
    attn = nn.MultiheadAttention(n_hidden, n_head)
    attn_out, attn_weight = attn(q, k, v)
    print("atten_out shape: ", attn_out.shape)
    print("atten_weight shape: ", attn_weight.shape)

    head_out = F.scaled_dot_product_attention(q, k, v)

    print("head_out shape: ", head_out.shape)
