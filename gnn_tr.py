import torch
from torch import nn, einsum
from torch_geometric.nn import global_mean_pool
from gnn_models import GNN
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# residual connection class and class to exclude cls token

class ResConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ExcCLS(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim = 1)

# normalization class

class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feed forward network classes

class DepthWiseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv(hidden_dim, hidden_dim, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
   
    def forward(self, x):
        h = x.shape[-2]
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = 1)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# multi-head self-attention

class MHSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResConnection(Norm(dim, MHSA(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ExcCLS(ResConnection(Norm(dim, FFN(dim, mlp_dim, dropout = dropout))))
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# main transformer class

class TR(nn.Module):
    def __init__(self, emb_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        patch_height, patch_width = pair(patch_size)
        
        patch_dim = channels * patch_height * patch_width
        
        n_patches = (emb_size // patch_height) * (1 // patch_width)
        
        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.localtr = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, emb):
        
        emb = emb.reshape(10,1,300,1)
        
        x = self.to_patch_emb(emb)
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_emb[:, :(n + 1)]
        
        x = self.dropout(x)

        x = self.localtr(x)

        return self.mlp_head(x[:, 0]), x[:, 0] 
    
class GNN_prediction(torch.nn.Module):

    def __init__(self, layer_number, emb_dim, jk = "last", dropout_prob= 0, pooling = "mean", gnn_type = "gin"):
        super(GNN_prediction, self).__init__()
        
        self.num_layer = layer_number
        self.drop_ratio = dropout_prob
        self.jk = jk
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of layers must be > 2.")

        self.gnn = GNN(layer_number, emb_dim, jk, dropout_prob, gnn_type = gnn_type)
        
        if pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Invalid pooling.")

        self.mult = 1
        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)
        
    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:0'), strict = False)
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("The arguments are unmatched!")

        node_embeddings = self.gnn(x, edge_index, edge_attr)
            
        pred_gnn = self.graph_pred_linear(self.pool(node_embeddings, batch))
        
        return pred_gnn, node_embeddings
        
        
if __name__ == "__main__":
    pass
