from perxeiver.components import *

from typing import Optional

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 num_latents,
                 latents_dim,
                 input_dim,
                 q_dim,
                 num_blocks,
                 num_layers,
                 
                 qk_out_dim: Optional[int] = None,
                 v_out_dim: Optional[int] = None,
                 
                 out_dim: Optional[int] = None,

                 num_attn_heads: int = 1,
                 
                 ffn_hidden_dim: int = 256,
                 

                 dropout: float = 0.0,

                 q_residual: bool = True,

                 num_experts: int = 10,
                 moe_top_k: Optional[int] = None
                 
                 ) -> None:
        
        super().__init__()
        
        self.encoder = Encoder(
            num_latents=num_latents,
            latents_dim=latents_dim,
            input_dim=input_dim,
            
            num_blocks=num_blocks,
            num_layers=num_layers,
            
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            
            num_cross_attn_heads=num_attn_heads,
            num_self_attn_heads=num_attn_heads,
            
            self_attn_ffn_hidden_dim=ffn_hidden_dim,
            cross_attn_ffn_hidden_dim=ffn_hidden_dim,
            
            dropout=dropout,
            cross_attn_dropout=dropout,
            self_attn_dropout=dropout,
            
            q_residual=q_residual,
            
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )
        
        self.decoder = Decoder(
            latents_dim=latents_dim,
            q_dim=q_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            num_heads=num_attn_heads,
            
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            proj_out_dim=out_dim,
            
            q_residual=q_residual,
            
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )
        
        
    def forward(self, 
                x: torch.Tensor, 
                q: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None,
                q_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        latents = self.encoder(x, kv_mask=kv_mask)
        out = self.decoder(q, latents, q_mask=q_mask)
        
        return out
    
    
if __name__ == "__main__":
    model = Model(
        num_latents=256,
        latents_dim=1024,
        input_dim=768,
        q_dim=512,
        num_blocks=4,
        num_layers=6,
        
        out_dim=768,
        
        num_attn_heads=8,
        ffn_hidden_dim=2048,
        
        dropout=0.0,
        
        q_residual=True,
        
        num_experts=5,
        moe_top_k=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())

    print(total_params)
    
    x = torch.randn(4, 1234, 768)
    kv_mask = torch.ones(4, 1234)
    q = torch.randn(4, 128, 512)
    q_mask = torch.ones(4, 128)
    
    out = model(x, q, kv_mask=kv_mask, q_mask=q_mask)
    
    print(out.shape)