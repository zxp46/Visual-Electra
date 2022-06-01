import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches

# TODO: ADD Downsampling function

def UpSampling(x, H, W):
        B, N, C = x.size()
        assert N == H*W
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = nn.PixelShuffle(2)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H*W)
        x = x.permute(0,2,1)
        return x, H, W

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, H, W, norm_layer=nn.LayerNorm):
        super().__init__()
        self.H = H
        self.W = W
        self.dim = dim
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = self.H
        W = self.W
        print("x",x.shape)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        print("Before Norm",x.shape)
        x = self.norm(x)

        return x

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, noise_dim=1024, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0., input_channel=3,patch_size=4,image_size=32,depth=7):#,device=device):
        super(Generator, self).__init__()

        #self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.mlp = nn.Linear(noise_dim, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2)+1, 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2+1, 384//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2+1, 384//16))
        
        print("Position",self.positional_embedding_1.shape)

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)

        self.PatchMerging3 = PatchMerging(dim*16, 32, 32)
        self.PatchMerging2 = PatchMerging(dim*4, 16, 16)
        self.PatchMerging1 = PatchMerging(dim*2, 8, 8)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))
        
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size//patch_size) ** 2
        self.patch_size = patch_size
        self.depth = depth

        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.class_embedding, std=0.2)
        self.patches = ImgPatches(input_channel, dim, self.patch_size)
        self.droprate = nn.Dropout(p=drop_rate)
    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1[:,1:,:]
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_2[:,1:,:]
        x = self.TransformerEncoder_encoder2(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_3[:,1:,:]

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        return x

    def forward_d(self, x):
        #TODO change CLS token
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding_1
        x = self.droprate(x)
        x = self.TransformerEncoder_encoder1(x)
        x = torch.cat(x[:,0,:],self.PatchMerging1(x[:,1:,:]))
        x += self.positional_embedding_2
        x = self.droprate(x)
        x = self.TransformerEncoder_encoder2(x)
        x = self.PatchMerging2(x)
        x += self.positional_embedding_3
        x = self.droprate(x)
        x = self.TransformerEncoder_encoder3(x)
        x = self.PatchMerging3(x)
        x = x.view(b, -1)
        x = self.norm(x)
        x = self.out(x[:, 0]).mean(dim=-1)
        return x

    def get_parameter_list(self):
        return [self.TransformerEncoder_encoder1, self.TransformerEncoder_encoder2, self.TransformerEncoder_encoder3],\
            [self.positional_embedding_1, self.positional_embedding_2, self.positional_embedding_3]


# TODO: Modify D to share parameters with G
class Discriminator(nn.Module):
    def __init__(self, image_size=32, patch_size=4, input_channel=3, num_classes=1,
                 dim=24, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0., transformer_blocks=[], positional_embeddings=[]):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size//patch_size) ** 2
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding_1, self.positional_embedding_2, self.positional_embedding_3 = positional_embeddings
        self.TransformerEncoder_encoder1, self.TransformerEncoder_encoder2, self.TransformerEncoder_encoder3 = transformer_blocks
        self.PatchMerging1 = PatchMerging(dim, 32, 32)
        self.PatchMerging2 = PatchMerging(dim*4, 16, 16)
        self.PatchMerging3 = PatchMerging(dim*16, 8, 8)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
                                      mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        # cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        # x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding1
        x = self.droprate(x)
        x = self.TransfomerEncoder1(x)
        x = self.PatchMerging1(x)
        x += self.positional_embedding2
        x = self.droprate(x)
        x = self.TransfomerEncoder2(x)
        x = self.PatchMerging2(x)
        x += self.positional_embedding_3
        x = self.droprate(x)
        x = self.TransfomerEncoder3(x)
        x = self.PatchMerging3(x)
        x = x.view(b, -1)
        x = self.norm(x)
        x = self.out(x.mean(dim=1))
        return x