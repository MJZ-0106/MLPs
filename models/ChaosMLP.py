# 2023.05.10-Main script for ChaosMLP model for tongue/ facial consititution image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 9, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
    }

default_cfgs = {
    'ChaosCAT_T': _cfg(crop_pct=0.9),
    'ChaosCAT_S': _cfg(crop_pct=0.9),
    'ChaosCAT_M': _cfg(crop_pct=0.9),
    'ChaosCAT_B': _cfg(crop_pct=0.875),
}

class Mlp(nn.Module):   # 一维卷积
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlplayer(nn.Module):  # 线性层
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChaosCAT(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.rho = 4
        self.h_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                )
        self.w_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, bias=True),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                )
        self.fc = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
         
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.fc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape    # B为batch size, C为卷积后输出特征的维度dim, H为输出特征的高度，W为输出特征的宽度
        
        h = self.h_conv(x)      # Conv 1×1,BN,ReLU
        w = self.w_conv(x)      # Conv 1×1,BN,ReLU
        c = self.fc(x)

        h_wight = self.fc(h)
        w_wight = self.fc(w)
        hh = torch.cat([h_wight*H*self.rho*(1-H), h], dim=1) # logistic Map of H
        ww = torch.cat([w_wight*W*self.rho*(1-W), w], dim=1) # logistic Map of W

        h = self.fc_h(hh)
        w = self.fc_w(ww)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)    # 平均池化输出
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)        # 混合sum
        x = self.proj_drop(x)    

        return x
        
class ChaosBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = ChaosCAT(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, embedding_dim // 2, kernel_size=(3, 3), # 输入通道数，输出维度，kernel size
                      stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 2, kernel_size=(3, 3), 
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, embedding_dim, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                         dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)

class PatchEmbedOverlapping(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, 
                norm_layer=nn.BatchNorm2d, groups=1, use_norm=True):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding= (padding, padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)      
        self.norm = norm_layer(embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # B, C, H, W
        x = self.norm(x)

        return x

class PatchEmbed(nn.Module):    # 2D Image to Patch Embedding
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)#.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_layer=nn.BatchNorm2d,use_norm=True):
        super().__init__()

        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_layer(out_embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # B, C, H, W
        x = self.norm(x)

        return x

def basic_blocks(dim, index, layers, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.,norm_layer=nn.BatchNorm2d, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ChaosBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer))
    blocks = nn.Sequential(*blocks)
    return blocks

class ChaosCATNet(nn.Module):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=10,
        embed_dims=None, transitions=None, mlp_ratios=None,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.BatchNorm2d, fork_feat=False,ds_use_norm=True,args=None): 
        super().__init__()
        
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0],norm_layer=norm_layer,use_norm=ds_use_norm)
        # self.tokenizer = ConvTokenizer(embedding_dim=embed_dims[0])
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=7, in_c=in_chans, embed_dim=embed_dims[0],norm_layer=norm_layer)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size, norm_layer=norm_layer, use_norm=ds_use_norm))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1]) 
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # x = self.tokenizer(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # print(x.shape)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        # print(x.shape)
        x = self.norm(x)
        cls_out = self.head(F.adaptive_avg_pool2d(x,output_size=1).flatten(1))
        return cls_out

def MyNorm(dim):
    return nn.GroupNorm(1, dim)    
    

@register_model
def ChaosMLP_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = ChaosCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ChaosCAT_T']
    return model

@register_model
def ChaosMLP_T1(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 4, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = ChaosCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ChaosCAT_T']
    return model

@register_model
def ChaosMLP_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = ChaosCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm, **kwargs)
    model.default_cfg = default_cfgs['ChaosCAT_S']
    return model

@register_model
def ChaosMLP_M(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = ChaosCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['ChaosCAT_M']
    return model

@register_model
def ChaosMLP_B(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 18, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = ChaosCATNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios,norm_layer=MyNorm,ds_use_norm=False, **kwargs)
    model.default_cfg = default_cfgs['ChaosCAT_B']
    return model
