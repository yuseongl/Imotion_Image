import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor

class PatchEmbed(nn.Module):
    '''Moduel for embeding as patch of image
    '''
    # def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
    def __init__(self, patch_size:int=16, in_chans:int=3, dim:int=768, norm_layer=None):
        self.patch_size = patch_size
        super().__init__()
        self.proj = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            # same as torch.flatten(2).transpose(1, 2)
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        x = self.norm(x)
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # # prepend the cls token to the input
        # x = torch.cat([cls_tokens, x], dim=1)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(768, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # for einops
        # qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class mlp_layer(nn.Module):
    def __init__(self, dim, mlp_dim, attn_drop):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Dropout(attn_drop),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(attn_drop)
        )

    def forward(self,x):
        identity = x
        x = self.mlp(x)
        return x + identity

class encoder_layer(nn.Module):
    def __init__(self, attention, dim, mlp_dim, attn_drop):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(dim),
            attention
        )
        
        self.mlp = mlp_layer(dim, mlp_dim, attn_drop)
        
        
    def forward(self, x):
        identity = x
        x = self.layer(x) + identity
        x = self.mlp(x)
        return x 
    
class encoder(nn.Module):
    def __init__(self, 
                    dim=768, 
                    mlp_dim=2048,  
                    heads_num=64, 
                    qkv_bias=False, 
                    qk_scale=None, 
                    attn_drop=0., 
                    proj_drop=0., 
                    patch_num=16, 
                    layers_num=6):
        super().__init__()
        self.PatchEmbeding = PatchEmbed(patch_num, in_chans=3, dim=768, norm_layer=None)
        self.pos_embedding = nn.Parameter(torch.randn(1,14**2+1,dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        attention = Attention(dim, heads_num, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.layers = nn.ModuleList([encoder_layer(attention, dim, mlp_dim, attn_drop)
                                        for _ in range(layers_num)])
        
    def forward(self,x):
        path_emb = self.PatchEmbeding(x)
        b,n,_ = path_emb.shape
        cls_token = repeat(self.cls_token,'1 1 d->b 1 d',b=b)
        
        pos_emb = self.pos_embedding[:,:n+1]
        
        x = torch.cat((cls_token,path_emb),dim=1)
        x = x + pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        return x
        
class vit(nn.Module):
    def __init__(self, 
                dim=768, 
                mlp_dim = 2048,
                num_classes=7, 
                heads_num=32, 
                qkv_bias=True, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0., 
                patch_num=16, 
                layers_num=8) -> None:
        super().__init__()
        self.encoder = encoder(dim, 
                            mlp_dim, 
                            heads_num, 
                            qkv_bias, 
                            qk_scale, 
                            attn_drop, 
                            proj_drop, 
                            patch_num, 
                            layers_num)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim,num_classes)
        )
    
    def forward(self,x):
        x = self.encoder(x)
        cls = x[:,0]
        out = self.fc_layer(cls)
        return out