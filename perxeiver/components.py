import torch
import torch.nn as nn

from typing import Optional, Tuple


class Embed2d(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 input_dim: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 ) -> None:
        super().__init__()

        # patches with overlap
        self.conv2d = nn.Conv2d(input_dim, embed_dim,
                                kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, E, H', W')
        x = self.conv2d(x)
        # (B, E, H', W') -> (B, E, L) -> (B, L, E)
        x = x.flatten(2).transpose(1, 2)

        return x


class Embed3d(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 input_dim: int,
                 kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 ) -> None:

        super().__init__()

        # patches with overlap
        self.conv3d = nn.Conv3d(input_dim, embed_dim,
                                kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, E, D', H', W')
        x = self.conv3d(x)
        # (B, E, D', H', W') -> (B, E, L) -> (B, L, E)
        x = x.flatten(2).transpose(1, 2)

        return x


class Linear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool = True,
                 num_experts: int = 10,
                 top_k: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.moe = num_experts > 1
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else num_experts // 2

        self.gate = nn.Sequential(
            nn.Linear(in_dim, num_experts, bias=True),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=bias) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.moe:
            return self.experts[0](x)

        gate = self.gate(x)

        # top-k gating, set rest to zero
        if self.top_k is not None:
            topk, indices = torch.topk(gate, self.top_k, dim=-1)
            gate = torch.zeros_like(gate)
            gate.scatter_(-1, indices, topk)
            gate = gate / gate.sum(-1, keepdim=True)

        x = torch.stack([expert(x) for expert in self.experts], dim=-1)

        x = (x * gate.unsqueeze(-2)).sum(-1)

        return x


class MultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 ) -> None:

        super().__init__()

        if qk_out_dim is None:
            qk_out_dim = q_dim

        if v_out_dim is None:
            v_out_dim = kv_dim

        if out_dim is None:
            out_dim = v_out_dim

        self.num_heads = num_heads
        self.qk_head_dim = qk_out_dim // num_heads
        self.v_head_dim = v_out_dim // num_heads

        self.q_proj = Linear(
            q_dim, qk_out_dim, num_experts=num_experts, top_k=moe_top_k)
        self.k_proj = Linear(kv_dim, qk_out_dim,
                             num_experts=num_experts, top_k=moe_top_k)
        self.v_proj = Linear(
            kv_dim, v_out_dim, num_experts=num_experts, top_k=moe_top_k)

        self.out_proj = Linear(v_out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.qk_head_dim ** 0.5)

    def forward(self,
                q: torch.Tensor,
                kv: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        q: torch.Tensor = self.q_proj(q)
        k: torch.Tensor = self.k_proj(kv)
        v: torch.Tensor = self.v_proj(kv)

        # split into heads
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.qk_head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.qk_head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.v_head_dim)

        # transpose to get dimensions [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention scores
        attn = q @ k.transpose(2, 3) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(
                attn_mask == 0, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)

        # apply dropout
        attn = self.dropout(attn)

        # apply weighted sum
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, 0)

        x = attn @ v

        # transpose to get dimensions [batch_size, seq_len, num_heads, head_dim]
        x = x.transpose(1, 2)

        # concat heads and project
        x = x.flatten(2)
        x = self.out_proj(x)

        return x, attn


class Mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float = 0.0,
                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 ) -> None:

        super().__init__()

        self.linear1 = Linear(in_dim, hidden_dim,
                              num_experts=num_experts, top_k=moe_top_k)
        self.activation = nn.GELU()
        self.linear2 = Linear(hidden_dim, out_dim,
                              num_experts=num_experts, top_k=moe_top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 ffn_hidden_dim: int,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 ) -> None:

        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.qkv_layer_norm = nn.LayerNorm(dim)

        self.attn = MultiheadAttention(
            q_dim=dim,
            kv_dim=dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            out_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = Mlp(dim, ffn_hidden_dim, dim, dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = x
        x = self.layer_norm(x)
        x, attn = self.attn(x, x, attn_mask)
        x = self.dropout(x)

        x = residual + x
        x = x + self.ffn(self.qkv_layer_norm(x))

        return x, attn


class CrossAttention(nn.Module):
    def __init__(self,
                 q_dim: int,
                 kv_dim: int,
                 ffn_hidden_dim: int,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 q_residual: bool = True,
                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)

        self.attn = MultiheadAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            out_dim=q_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = Mlp(q_dim, ffn_hidden_dim, q_dim, dropout)

        self.q_residual = q_residual

    def forward(self,
                q: torch.Tensor,
                kv: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = q
        q = self.q_layer_norm(q)
        kv = self.kv_layer_norm(kv)
        q, attn = self.attn(q, kv, attn_mask)
        q = self.dropout(q)

        if self.q_residual:
            q = residual + q

        q = q + self.ffn(self.qkv_layer_norm(q))

        return q, attn


class Encoder(nn.Module):
    def __init__(self,
                 num_latents: int,
                 latents_dim: int,
                 input_dim: int,
                 num_blocks: int,
                 num_layers: int,

                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,

                 num_self_attn_heads: int = 1,
                 num_cross_attn_heads: int = 1,

                 self_attn_ffn_hidden_dim: int = 2048,
                 cross_attn_ffn_hidden_dim: int = 2048,

                 dropout: float = 0.0,
                 self_attn_dropout: float = 0.0,
                 cross_attn_dropout: float = 0.0,

                 q_residual: bool = True,

                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None

                 ) -> None:

        super().__init__()

        self.num_blocks = num_blocks

        self.latents = nn.Parameter(torch.randn(num_latents, latents_dim))

        self.cross_attn = CrossAttention(
            q_dim=latents_dim,
            kv_dim=input_dim,
            ffn_hidden_dim=cross_attn_ffn_hidden_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
            attn_dropout=cross_attn_dropout,
            q_residual=q_residual,

            num_experts=num_experts,
            moe_top_k=moe_top_k
        )

        self.self_attn_layers = nn.ModuleList([
            SelfAttention(
                dim=latents_dim,
                ffn_hidden_dim=self_attn_ffn_hidden_dim,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim,
                num_heads=num_self_attn_heads,
                dropout=dropout,
                attn_dropout=self_attn_dropout,
                num_experts=num_experts,
                moe_top_k=moe_top_k
            ) for _ in range(num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = x.shape[0]

        if kv_mask is not None:
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(1)

        latents, attn = self.cross_attn(
            self.latents.repeat(batch_size, 1, 1),
            x,
            kv_mask
        )

        for _ in range(self.num_blocks):
            for self_attn_layer in self.self_attn_layers:
                latents, attn = self_attn_layer(latents)

        return latents


class Decoder(nn.Module):
    def __init__(self,
                 latents_dim: int,
                 q_dim: int,
                 ffn_hidden_dim: int,
                 num_heads: int = 1,
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 proj_out_dim: Optional[int] = None,
                 q_residual: bool = True,
                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 ) -> None:
        super().__init__()

        self.cross_attn = CrossAttention(
            q_dim=q_dim,
            kv_dim=latents_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            num_heads=num_heads,
            q_residual=q_residual,
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )

        self.proj = Linear(q_dim, proj_out_dim, num_experts=num_experts,
                           top_k=moe_top_k) if proj_out_dim is not None else nn.Identity()

    def forward(self,
                q: torch.Tensor,
                latents: torch.Tensor,
                q_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        if q_mask is not None:
            q_mask = q_mask.unsqueeze(1).unsqueeze(1).transpose(-2, -1)

        x, attn = self.cross_attn(q, latents, q_mask)
        x = self.proj(x)
        return x


if __name__ == '__main__':
    # test model
    model = Encoder(
        num_latents=256,
        latents_dim=2048,
        input_dim=4096,
        num_blocks=4,
        num_layers=4,
        num_self_attn_heads=8,
        num_cross_attn_heads=8,
        self_attn_ffn_hidden_dim=4096,
        cross_attn_ffn_hidden_dim=4096,
        dropout=0.0,
        self_attn_dropout=0.0,
        cross_attn_dropout=0.0,
        q_residual=True,
        num_experts=7,
        moe_top_k=3
    ).to('mps')

    # calculate parameter number
    total_params = sum(p.numel() for p in model.parameters())

    print(total_params)

    x = torch.randn(4, 1234, 4096).to('mps')
    kv_mask = torch.ones(4, 1234).to('mps')

    latents = model(x, kv_mask)

    print(model)

    print(latents.shape)
