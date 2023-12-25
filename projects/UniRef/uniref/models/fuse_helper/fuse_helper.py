import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

try:
    from .flash_attention import FlashCrossAttention  # v2
    from flash_attn.bert_padding import unpad_input, pad_input
except:
    print("FlashAttention is not installed.")


class UniMultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, head_dim, num_heads, dropout=0.1, cfg=None):
        super(UniMultiHeadAttention, self).__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.embed_dim = head_dim * num_heads
        self.q_dim = q_dim 
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.inner_attn = FlashCrossAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous() # [bs, seq, num_heads, head_dim]

    def _pad(self, tensor, max_len):
        # pad tensor [bs, len, c] or mask [bs, len]
        assert tensor.dim() == 3 or tensor.dim() == 2
        if tensor.dim() == 3:  # tensor
            bs, len, c = tensor.shape
            pad_tensor = torch.zeros((bs, max_len-len, c), dtype=tensor.dtype, device=tensor.device)
            new_tensor = torch.cat([tensor, pad_tensor], dim=1)  
        if tensor.dim() == 2:  # mask
            bs, len = tensor.shape
            pad_tensor = torch.zeros((bs, max_len-len), dtype=tensor.dtype, device=tensor.device)
            new_tensor = torch.cat([tensor, pad_tensor], dim=1)
        return new_tensor

    def forward(self, q, k, v=None, attention_mask=None):
        # q: [b, q_len, c]
        # k: [b, k_len, c]
        # v: [b, k_len, c]
        # attention_mask (LongTensor): [b, k_len], valid locations are 1
        bsz, q_len, q_dim = q.size()
        bsz, k_len, k_dim = k.size()

        query_states = self.q_proj(q)  # q, [b, q_len, c]
        key_states = self.k_proj(k)    # k, [b, k_len, c]
        if v is None:
            value_states = self.v_proj(k) # [b, k_len, c]
        else:
            value_states = self.v_proj(v) # [b, k_len, c]

        # flash_attn_varlen_kvpacked_func
        # q: (total_q, nheads, headdim)
        # kv: (total_k, 2, nheads_k, headdim)
        # cu_seqlens_q: (batch_size + 1,), dtype torch.int32. 
        # cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        # max_seqlen_q: int. 
        # max_seqlen_k: int

        data_dtype = query_states.dtype
        # reshape, [bs, max_len, num_heads, head_dim]
        query_states = self._shape(query_states, q_len, bsz).to(torch.bfloat16)
        key_states = self._shape(key_states, k_len, bsz).to(torch.bfloat16)
        value_states = self._shape(value_states, k_len, bsz).to(torch.bfloat16)
        # query
        max_seqlen_q = query_states.shape[1]
        cu_seqlens_q = torch.arange(0, (bsz + 1) * max_seqlen_q, step=max_seqlen_q, dtype=torch.int32,
                                          device=query_states.device)  # e.g. [0, 10, 20] when seqlen=10
        q_states = query_states.flatten(0, 1)  # [bsz * q_len, nheads, head_dim]
        # kv
        kv_states = torch.stack([key_states, value_states], dim=2)  # [bs, k_len, 2, num_heads, head_dim]
        kv_states = kv_states.flatten(2)  # [bs, k_len, 2 * num_heads * head_dim]
        kv_unpad, indices, cu_seqlens_k, max_seqlen_k = unpad_input(kv_states, attention_mask)
        kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=self.num_heads)  # [total_k, 2, nheads, head_dim]

        # flash_attn
        out = self.inner_attn(q_states, kv_unpad, cu_seqlens=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen=max_seqlen_q, max_seqlen_k=max_seqlen_k)  # [total_q, nheads, head_dim]
        out = out.reshape(bsz, q_len, self.num_heads, self.head_dim).flatten(2)
        out = out.to(data_dtype)
        out = self.out_proj(out)
        return out


