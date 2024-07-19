# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...common import LayerNorm2d
from ...ImageEncoder import AdapterBlock, Block, LoraBlock
from .common import LayerNorm2d, MLPBlock, Adapter, SpatialSelfattentionBlock, ChannelSelfattentionBlock, Down, Up, SingleDown, SingleUp


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        # args,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        low_image_size:int = (64, 64)
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of
             ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        # self.args = args

        self.low_image_size = low_image_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // patch_size, 1024 // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        block_class = AdapterBlock
        # if args.mod == 'sam_adpt':
        #     block_class = AdapterBlock
        # elif args.mod == 'sam_lora':
        #     block_class = LoraBlock
        # else:
        #     block_class = Block

        for i in range(depth):
            block = block_class(
                # args=self.args,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0, # 当i不是[2,5,8,11]中的数时，window_size等于14，否则，window_size等于0
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        base_dim = 48

        self.samct_cnn_embed = CNNEmbed(in_chans=in_chans, embed_dim=base_dim)

        self.samct_cnndown1 = Down(base_dim, 2 * base_dim)
        self.samct_trans2cnn1 = Trans2CNN(dimcnn=2 * base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size // 2)

        self.samct_cnndown2 = Down(2 * base_dim, 4 * base_dim)  # 192
        self.samct_trans2cnn2 = Trans2CNN(dimcnn=4 * base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size // 4)

        self.samct_cnndown3 = Down(4 * base_dim, 8 * base_dim)  # 384
        self.samct_trans2cnn3 = Trans2CNN(dimcnn=8 * base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size // 8)

        self.samct_cnndown4 = Down(8 * base_dim, 16 * base_dim)  # 768
        self.samct_trans2cnn4 = Trans2CNN(dimcnn=16 * base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size // 16)

        self.samct_up1 = Up(16*base_dim, 8*base_dim, bilinear=False) # 32*32
        self.samct_up2 = Up(8*base_dim, 4*base_dim, bilinear=False) # 64*64
        self.samct_up3 = Up(4*base_dim, 2*base_dim, bilinear=False) # 128*128
        self.samct_up4 = Up(2*base_dim, base_dim, bilinear=False) # 256*256
        self.samct_neck = nn.Conv2d(base_dim, out_chans//8, kernel_size=1, bias=False,)

        # egi
        self.egi = EGI(256,2)

        self.lineare0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.lineare1 = nn.Conv2d(48, 1, kernel_size=3, stride=1, padding=1)
        self.lineare2 = nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1)
        self.lineare3 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1)
        self.lineare4 = nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # B C H W (b, 3, 256, 256)

        cnnx = self.samct_cnn_embed(x)  # (b c 256 256) (1, 3, 256, 256) -> (1, 48, 256, 256)
        transx = self.patch_embed(x) # (B, C, H, W) -> (B, H, W, C) ,(b, 16, 16, 768)

        if self.pos_embed is not None:
            # resize position embedding to match the input
            new_abs_pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2), # (B, H, W, C) -> (B, C, H, W)
                size=(transx.shape[1], transx.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
            transxi = transx + new_abs_pos # (B, H, W, C)


        transx = self.blocks[0](transx) # (b, 16, 16, 768)
        transx = self.blocks[1](transx) # (b, 16, 16, 768)
        transxi = self.blocks[2](transx) # (b, 16, 16, 768)
        cnnx1 = self.samct_cnndown1(cnnx) # (b c1 128 128) (b, 96, 128, 128)
        cnnx1 = self.samct_trans2cnn1(transxi, cnnx1) + cnnx1 # (b, 96, 128, 128)

        transx = self.blocks[3](transxi)
        transx = self.blocks[4](transx)
        transxi = self.blocks[5](transx)
        cnnx2 = self.samct_cnndown2(cnnx1)  # (b c2 64 64) (b 192 64 64)
        cnnx2 = self.samct_trans2cnn2(transxi, cnnx2) + cnnx2 # (b 192 64 64)

        transx = self.blocks[6](transxi)
        transx = self.blocks[7](transx)
        transxi = self.blocks[8](transx)
        cnnx3 = self.samct_cnndown3(cnnx2)  # (b c3 32 32)  (b 384 32 32)
        cnnx3 = self.samct_trans2cnn3(transxi, cnnx3) + cnnx3 # (b 384 32 32)

        transx = self.blocks[9](transxi)
        transx = self.blocks[10](transx)
        transxi = self.blocks[11](transx)
        cnnx4 = self.samct_cnndown4(cnnx3)  # (b c4 16 16) (b 768 16 16)
        cnnx4 = self.samct_trans2cnn4(transxi, cnnx4) + cnnx4 # (b 768 16 16)

        transx = transxi.permute(0, 3, 1, 2) # (b, 768, 16, 16)

        x = cnnx4 # (b 768 16 16)
        e4 = self.samct_up1(x, cnnx3) # (b, 384, 32, 32)
        e3 = self.samct_up2(e4, cnnx2) # (b, 192, 64, 64)
        e2 = self.samct_up3(e3, cnnx1) # (b, 96, 128, 128)
        e1 = self.samct_up4(e2, cnnx) # (b, 48, 256, 256)
        e0 = self.samct_neck(e1) # (b, 32, 256, 256)

        transx = self.neck(transx)  # (b, 256, 16, 16)

        #  Edge guide Interactive
        transx = self.egi(transx, e0)

        # edge predict
        edge0 = F.interpolate(self.lineare0(e0), size=self.low_image_size, mode='bilinear', align_corners=True) # B1
        edge1 = F.interpolate(self.lineare1(e1), size=self.low_image_size, mode='bilinear', align_corners=True) # B2
        edge2 = F.interpolate(self.lineare2(e2), size=self.low_image_size, mode='bilinear', align_corners=True) # B3
        edge3 = F.interpolate(self.lineare3(e3), size=self.low_image_size, mode='bilinear', align_corners=True) # B2
        edge4 = F.interpolate(self.lineare4(e4), size=self.low_image_size, mode='bilinear', align_corners=True) # B3



        return transx, edge4, edge3, edge2, edge1, edge0

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class Trans2CNN(nn.Module):
    """Achieved by a spatial attention module"""

    def __init__(
        self,
        dimtrans: int,
        dimcnn: int,
        cnn_patch_size: int = 2,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.window_size = cnn_patch_size
        self.sa = SpatialAttention()
        self.fc = nn.Conv2d(dimcnn, dimtrans, kernel_size=1, bias=False)
        self.scale = dimtrans**0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_trans: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x_trans.shape
        trans = rearrange(x_trans, 'b H W c->b c H W')
        sa = self.sa(trans) # b 1 H W
        sa = rearrange(sa, 'b g H W -> (b H W) g') # (BHW 1)
        q = rearrange(trans, 'b c H W -> (b H W) c').unsqueeze(dim=1) # (bHW 1 c)
        k = rearrange(self.fc(x_cnn), 'b c (H m) (W n) -> (b H W) (m n) c', m=self.window_size, n=self.window_size) # (bHW kk c)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (bHW 1 kk)
        attn = attn.softmax(dim=-1)
        attn = 0.5 + self.sigmoid(attn - 1/(self.window_size*self.window_size*1.0))  # BHW 1 kk
        attn = attn * sa[:, :, None]
        attn = rearrange(attn, '(b H W) g (m n) -> b g (H m) (W n)', m=self.window_size, H=H, W=W)
        out = x_cnn * attn
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out) # b 1 h w
        return self.sigmoid(out)

class EGI(nn.Module):
    def __init__(self,in_channel=256, ratio=2):
        super(EGI, self).__init__()

        self.conv_query = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_key = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, sod, edge):
        if edge.size()[2:] != sod.size()[2:]:
            edge = F.interpolate(edge, size=sod.size()[2:], mode='bilinear', align_corners=True)
        bz,c,h,w=sod.shape

        edge_q = self.conv_query(edge).view(bz, -1, h * w).permute(0, 2, 1)
        edge_k = self.conv_key(edge).view(bz, -1, h * w)
        mask = torch.bmm(edge_q, edge_k)  # bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(sod).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1))  # bz, c, hw
        feat = feat.view(bz, c, h, w)
        out = sod + feat * sod

        return out

class CNNEmbed(nn.Module):
    """
    Image to CNN Embedding.
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.csb = ChannelSelfattentionBlock(in_chans, embed_dim)
        self.ssb = SpatialSelfattentionBlock(in_chans, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = self.csb(x)
        out = self.ssb(x, xc)
        return out