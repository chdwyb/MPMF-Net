import torch
import torch.nn as nn
import torch.nn.functional as F
from INR import INR
from base_modules import RLN, PatchEmbed, PatchUnEmbed, Attention, Mlp


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=RLN, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super(TransformerBlock, self).__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=RLN, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# Multi-dimension Feature Interaction Block (MFIB)
class MFIB(nn.Module):
    def __init__(self, dim=32):
        super(MFIB, self).__init__()
        self.scale_h = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.scale_w = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.scale_c = nn.Parameter(torch.ones((1)), requires_grad=True)

        self.norm_u = RLN(dim)
        self.norm_d = RLN(dim)

        self.pconv_qkv_u = nn.Conv2d(dim, dim * 2, 1)
        self.pconv_qkv_d = nn.Conv2d(dim, dim * 2, 1)

    def forward(self, x_u, x_d):
        x_d_norm, rescale_d, bias_d = self.norm_d(x_d)
        x_u_norm, rescale_u, bias_u = self.norm_u(x_u)

        q_u, v_u = self.pconv_qkv_u(x_u_norm).chunk(2, dim=1)
        k_d, v_d = self.pconv_qkv_d(x_d_norm).chunk(2, dim=1)

        q_u_h, q_u_w, q_u_c = q_u.chunk(3, dim=1)
        k_d_h, k_d_w, k_d_c = k_d.chunk(3, dim=1)
        v_u_h, v_u_w, v_u_c = v_u.chunk(3, dim=1)
        v_d_h, v_d_w, v_d_c = v_d.chunk(3, dim=1)

        # h
        q_u_h = q_u_h.permute(0, 3, 2, 1)
        k_d_h = k_d_h.permute(0, 3, 1, 2)
        v_u_h = v_u_h.permute(0, 3, 2, 1)
        v_d_h = v_d_h.permute(0, 3, 2, 1)
        attn_h = torch.matmul(q_u_h, k_d_h) * self.scale_h
        attn_h = torch.softmax(attn_h, dim=-1)
        x_d2u_h = torch.matmul(attn_h, v_d_h)
        x_u2d_h = torch.matmul(attn_h.transpose(-1, -2), v_u_h)

        # w
        q_u_w = q_u_w.permute(0, 2, 3, 1)
        k_d_w = k_d_w.permute(0, 2, 1, 3)
        v_u_w = v_u_w.permute(0, 2, 3, 1)
        v_d_w = v_d_w.permute(0, 2, 3, 1)
        attn_w = torch.matmul(q_u_w, k_d_w) * self.scale_w
        x_d2u_w = torch.matmul(torch.softmax(attn_w, dim=-1), v_d_w)
        x_u2d_w = torch.matmul(torch.softmax(attn_w.transpose(-1, -2), dim=-1), v_u_w)

        # c
        b, c, h, w = q_u_c.shape
        q_u_c = q_u_c.reshape(b, c, h * w)
        k_d_c = k_d_c.reshape(b, c, h * w).permute(0, 2, 1)
        v_u_c = v_u_c.reshape(b, c, h * w)
        v_d_c = v_d_c.reshape(b, c, h * w)
        attn_c = torch.matmul(q_u_c, k_d_c) * self.scale_c
        x_d2u_c = torch.matmul(torch.softmax(attn_c, dim=-1), v_d_c).reshape(b, c, h, w)
        x_u2d_c = torch.matmul(torch.softmax(attn_c, dim=-1), v_u_c).reshape(b, c, h, w)

        x_d2u = torch.cat([x_d2u_h.permute(0, 3, 2, 1), x_d2u_w.permute(0, 3, 1, 2), x_d2u_c], dim=1)
        x_u2d = torch.cat([x_u2d_h.permute(0, 3, 2, 1), x_u2d_w.permute(0, 3, 1, 2), x_u2d_c], dim=1)

        x_u = x_d2u * rescale_u + bias_u + x_u

        x_d = x_u2d * rescale_d + bias_d + x_d

        return x_u + x_d


# Multi-axis Prompts Learning Block (MPLB)
class MPLB(nn.Module):
    def __init__(self, dim, size=(16, 16), bias=False):
        super(MPLB, self).__init__()

        partial_dim = int(dim // 3)

        self.hw = nn.Parameter(torch.ones(1, partial_dim, size[0], size[1]), requires_grad=True)
        self.conv_hw = nn.Conv2d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.ch = nn.Parameter(torch.ones(1, 1, partial_dim, size[0]), requires_grad=True)
        self.conv_ch = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.cw = nn.Parameter(torch.ones(1, 1, partial_dim, size[1]), requires_grad=True)
        self.conv_cw = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.conv_4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        input_ = x
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        # hw
        x1 = x1 * self.conv_hw(F.interpolate(self.hw, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ch
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_ch(
            F.interpolate(self.ch, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # cw
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_cw(
            F.interpolate(self.cw, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_4(x) + input_

        return x


# Multi-axis Prompting and Multi-dimension Fusion Network (MPMF-Net)
class MPMFNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=(24, 48, 96, 48, 24),
                 mlp_ratios=(2., 4., 4., 2., 2.),
                 depths=(4, 4, 4, 9, 9, 9),
                 num_heads=(2, 4, 6, 1, 1),
                 attn_ratio=(1 / 4, 1 / 2, 3 / 4, 0, 0),
                 conv_type=('DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'),
                 norm_layer=(RLN, RLN, RLN, RLN, RLN)):
        super(MPMFNet, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_embed_inr = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer_inr = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.inr = INR(dim=embed_dims[0])

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans * 2, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2] -1,
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.layer3_prompts = MPLB(embed_dims[2], size=(16, 16))
        self.layer3_mixer = BasicLayer(network_depth=sum(depths), dim=embed_dims[2] * 2, depth=1,
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        self.layer3_reduce_channel = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], 1)

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = MFIB(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3]-1,
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.layer4_prompts = MPLB(embed_dims[3], size=(32, 32))
        self.layer4_last = BasicLayer(network_depth=sum(depths), dim=embed_dims[3]*2, depth=1,
                                      num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                      norm_layer=norm_layer[3], window_size=window_size,
                                      attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[2])
        self.layer4_reduce_channel = nn.Conv2d(embed_dims[3] * 2, embed_dims[3], 1)

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = MFIB(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4]-1,
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.layer5_prompts = MPLB(embed_dims[4], size=(64, 64))
        self.layer5_last = BasicLayer(network_depth=sum(depths), dim=embed_dims[4]*2, depth=1,
                                      num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                      norm_layer=norm_layer[4], window_size=window_size,
                                      attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])
        self.layer5_reduce_channel = nn.Conv2d(embed_dims[4] * 2, embed_dims[4], 1)

        self.refinement = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[5],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)

        x_prompts = self.layer3_prompts(x)
        x = torch.cat((x, x_prompts), dim=1)
        x = self.layer3_mixer(x)
        x = self.layer3_reduce_channel(x)

        x = self.patch_split1(x)

        x = self.fusion1(x, self.skip2(skip2))
        x = self.layer4(x)

        x_prompts = self.layer4_prompts(x)
        x = torch.cat((x, x_prompts), dim=1)
        x = self.layer4_last(x)
        x = self.layer4_reduce_channel(x)

        x = self.patch_split2(x)

        x = self.fusion2(x, self.skip1(skip1))
        x = self.layer5(x)

        x_prompts = self.layer5_prompts(x)
        x = torch.cat((x, x_prompts), dim=1)
        x = self.layer5_last(x)
        x = self.layer5_reduce_channel(x)

        x = self.refinement(x)
        x = self.patch_unembed(x)
        return x

    def inr_process(self, x):
        x_inr = F.interpolate(x, scale_factor=0.25)
        x_inr = self.patch_embed_inr(x_inr)
        x_inr = self.layer_inr(x_inr)
        x_inr = self.inr(x_inr)
        x = torch.cat((x, F.interpolate(x_inr, scale_factor=4)), dim=1)
        return x, x_inr

    def forward(self, x):
        input_ = x
        H, W = x.shape[2:]

        x, x_inr = self.inr_process(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * input_ - B + input_
        x = x[:, :, :H, :W]
        return x, x_inr


if __name__ == '__main__':
    x = torch.randn((1, 3, 64, 64)).cuda()
    net = MPMFNet().cuda()
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.4f")
    print(flops, params)