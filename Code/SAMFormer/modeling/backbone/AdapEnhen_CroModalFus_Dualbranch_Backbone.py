import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from timm.models.layers import trunc_normal_
import math
from detectron2.layers import Conv2d, get_norm, ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import BasicBlock, BasicStem, BottleneckBlock, ResNet, DeformBottleneckBlock

class ConcatFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuse_conv = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        fused = torch.cat([x1, x2], dim=1)
        fused = self.fuse_conv(fused)
        fused = self.norm(fused)
        fused = self.relu(fused)
        return fused

class CrossPath(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=None,
                 attention_operation="Modality_Bias_Embed_CrossAttention", ds_shape=None, spatial_down_ratio=None):
        super().__init__()
        self.ds_shape = ds_shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_down_ratio = spatial_down_ratio  # 保存下采样比例用于上采样

        # 初始化上采样层
        if self.spatial_down_ratio is not None and self.spatial_down_ratio > 1:
            self.upsample = nn.Upsample(scale_factor=spatial_down_ratio, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

        if attention_operation == "Modality_Bias_Embed_CrossAttention":
            self.cross_attn = Modality_Bias_Embed_CrossAttention(input_dim, num_heads=num_heads)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_operation}")

        # 调整end_proj的维度以匹配output_dim
        self.end_proj1 = nn.Linear(input_dim, output_dim)
        self.end_proj2 = nn.Linear(input_dim, output_dim)
        self.norm1 = nn.LayerNorm(output_dim)  # 同步调整LayerNorm的维度
        self.norm2 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

        # 初始化投影层
        init.kaiming_normal_(self.end_proj1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.end_proj2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        B, N, C = x1.shape
        # 不需要进行liear embedding，因为不需要压缩特征的维度
        v1, v2 = self.cross_attn(x1, x2)
        # print("Cross attention shape:", v1.shape, v2.shape)  # 打印交叉注意力后的形状

        out_x1 = self.norm1(self.relu(self.end_proj1(v1)))  # [B, N, output_dim]
        # print("End proj1 shape:", out_x1.shape)  # 打印最终投影后的形状
        out_x2 = self.norm2(self.relu(self.end_proj2(v2)))
        # print("End proj2 shape:", out_x2.shape)  # 打印最终投影后的形状

        # 将序列恢复为空间特征并上采样
        H_ds, W_ds = self.ds_shape
        out_x1 = out_x1.transpose(1, 2).reshape(B, self.output_dim, H_ds, W_ds)
        # print("Reshape x1 shape:", out_x1.shape)  # 打印重塑后的形状
        out_x2 = out_x2.transpose(1, 2).reshape(B, self.output_dim, H_ds, W_ds)
        # print("Reshape x2 shape:", out_x2.shape)  # 打印重塑后的形状

        # 上采样到原始分辨率
        out_x1 = self.upsample(out_x1)
        # print("Upsample x1 shape:", out_x1.shape)  # 打印上采样后的形状
        out_x2 = self.upsample(out_x2)
        # print("Upsample x2 shape:", out_x2.shape)  # 打印上采样后的形状

        return out_x1, out_x2

class ChannelEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(out_channels // reduction),  # 添加BN
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, padding=1,
                      groups=out_channels // reduction),
            nn.BatchNorm2d(out_channels // reduction),  # 添加BN
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1))
        self.norm = norm_layer(out_channels)

        # 初始化
        for m in self.channel_embed:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = self.residual(x)
        # print("Residual shape:", residual.shape)  # 打印残差连接的形状
        x = self.channel_embed(x)
        # print("Channel embed shape:", x.shape)  # 打印通道嵌入后的形状
        out = self.norm(residual + x)
        # print("Channel embed output shape:", out.shape)  # 打印通道嵌入模块的输出形状
        return out

class Modality_Bias_Embed_CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 应能被 num_heads {num_heads} 整除。"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义线性变换层
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)


        self.beta_1to2 = nn.Parameter(torch.full((num_heads, 1, 1), 0.2))
        self.beta_2to1 = nn.Parameter(torch.full((num_heads, 1, 1), 0.1))

        self._init_weights()

    def _init_weights(self):
        for m in [self.q1, self.kv1, self.q2, self.kv2]:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.normal_(m.bias, std=1e-6)

    def forward(self, x1, x2):
        B, N, C = x1.shape

        # 生成Q/K/V
        q1 = self.q1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1, v1 = self.kv1(x1).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2 = self.q2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2, v2 = self.kv2(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 计算注意力得分并添加模态偏差
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn1 += self.beta_1to2  # 添加模态偏差（广播到[B, num_heads, N, N]）
        attn1 = attn1.softmax(dim=-1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn2 += self.beta_2to1  # 添加模态偏差（广播到[B, num_heads, N, N]）
        attn2 = attn2.softmax(dim=-1)

        # 输出计算
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        return x1, x2
class AFEM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=4):
        super(AFEM, self).__init__()
        self.in_channels = in_channels

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, 2 * in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(2 * in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_channels // reduction_ratio, 2, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.channel_mlp:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.zeros_(m.bias)
        for m in self.spatial_conv:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, xR, xL):
        B, C, H, W = xR.shape

        # ================= 通道注意力阶段 =================
        # Step 1-5: 生成通道权重
        x_cat = torch.cat([xR, xL], dim=1)
        gmp = torch.amax(x_cat, dim=[2, 3], keepdim=True)
        gap = torch.mean(x_cat, dim=[2, 3], keepdim=True)
        y = torch.cat([gmp, gap], dim=1)
        wc = self.channel_mlp(y)
        wc_r, wc_l = torch.split(wc, [self.in_channels] * 2, dim=1)

        # 应用通道注意力
        xR_c_att = xR * wc_r  # [B,C,H,W]
        xL_c_att = xL * wc_l

        # ================= 空间注意力阶段 =================
        # 使用通道增强后的特征作为输入
        spatial_cat = torch.cat([xR_c_att, xL_c_att], dim=1)  # [B,2C,H,W]
        ws = self.spatial_conv(spatial_cat)  # [B,2,H,W]
        ws_r, ws_l = torch.split(ws, 1, dim=1)  # 各[B,1,H,W]

        # ================= 特征融合 =================
        xR_out = xR + xR_c_att * ws_r  # 原始 + (通道增强 * 空间权重)
        xL_out = xL + xL_c_att * ws_l

        return xR_out, xL_out

class CMFM(nn.Module):
    def __init__(self, dim, num_heads=8, norm_layer=nn.BatchNorm2d,
                 attention_operation="Modality_Bias_Embed_CrossAttention", spatial_down_ratio=4, dim_down=8):
        super().__init__()

        self.dim = dim
        self.dim_down = dim // dim_down

        # 空间下采样和维度下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(dim, self.dim_down, kernel_size=3, stride=spatial_down_ratio, padding=1),
            nn.BatchNorm2d(self.dim_down),
            nn.ReLU(inplace=True)
        )

        # 交叉路径
        self.cross = CrossPath(
            input_dim=self.dim_down,
            output_dim=dim,
            num_heads=num_heads,
            attention_operation=attention_operation,
            ds_shape=None,
            spatial_down_ratio=spatial_down_ratio
        )

        self.channel_emb = ChannelEmbed(
            in_channels=dim * 4,
            out_channels=dim,
            norm_layer=norm_layer
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        # 空间下采样
        x1_down = self.downsample(x1)  # [B, dim_down, H_ds, W_ds]
        x2_down = self.downsample(x2)
        H_ds, W_ds = x1_down.shape[2], x1_down.shape[3]

        # 更新cross模块的ds_shape
        self.cross.ds_shape = (H_ds, W_ds)

        # 展平处理
        x1_flat = x1_down.flatten(2).transpose(1, 2)  # [B, H_ds*W_ds, dim_down]
        x2_flat = x2_down.flatten(2).transpose(1, 2)

        # 交叉路径处理（自动包含上采样）
        x1_out, x2_out = self.cross(x1_flat, x2_flat)  # [B, C, H, W]

        # 合并特征
        merge = torch.cat([x1, x1_out, x2, x2_out], dim=1)  # [B, 4C, H, W]
        merge = self.channel_emb(merge)  # [B, C, H, W]

        return merge


class AdapEnhen_CroModalFus_Dualbranch_Backbone(Backbone):
    def __init__(self, stem_R, stem_L, stages, num_classes=None, out_features=None,
                 freeze_at=0, fuse_stages=None, enhance_module="AFEM",
                 fusion_module="CMFM", attention_module="Modality_Bias_Embed_CrossAttention",
                 alternate_fusion_module="ConcatFusion"):  # 新增备用模块参数
        super().__init__()
        self.num_classes = num_classes
        self.fuse_stages = fuse_stages if fuse_stages is not None else []
        self.alternate_fusion_module = alternate_fusion_module  # 保存备用模块类型


        import copy
        self.branch_R = ResNet(stem_R, stages, out_features=out_features, freeze_at=freeze_at)
        self.branch_L = ResNet(copy.deepcopy(stem_L), copy.deepcopy(stages), out_features=out_features, freeze_at=freeze_at)

        self.enhance_modules = nn.ModuleList()
        for stage_idx in range(4):
            in_channels = self.branch_R._out_feature_channels[f"res{stage_idx + 2}"]
            if enhance_module == "AFEM":
                module = AFEM(in_channels)
            else:
                raise ValueError(f"Unsupported enhance module: {enhance_module}")
            self.enhance_modules.append(module)

        self.fusion_modules = nn.ModuleDict()
        for stage in [2,3,4,5]:
            stage_name = f"res{stage}"
            in_channels = self.branch_R._out_feature_channels[stage_name]

            if stage in self.fuse_stages:
                if fusion_module == "CMFM":
                    module = CMFM(
                        dim=in_channels,
                        attention_operation=attention_module,
                        num_heads=8
                    )
                else:
                    raise ValueError(f"Unsupported fusion module: {fusion_module}")
            else:
                # 使用备用融合模块
                if self.alternate_fusion_module == "ConcatFusion":
                    module = ConcatFusion(in_channels)
                else:
                    raise ValueError(f"Unsupported alternate fusion module: {self.alternate_fusion_module}")

            self.fusion_modules[stage_name] = module

        self._out_features = out_features if out_features is not None else ["res5"]
        self._out_feature_channels = self.branch_R._out_feature_channels
        self._out_feature_strides = self.branch_R._out_feature_strides

    def forward(self, x):
        assert x.dim() == 4, "Input must be 4D tensor"
        xR = x[:, :3, :, :]
        xL = x[:, 3:, :, :]

        # 处理Stem
        xR = self.branch_R.stem(xR)
        xL = self.branch_L.stem(xL)

        outputs = {}
        for stage_idx in range(4):  # 遍历res2到res5
            stage_name = f"res{stage_idx + 2}"

            xR = self.branch_R.stages[stage_idx](xR)
            xL = self.branch_L.stages[stage_idx](xL)

            xR_prime, xL_prime = self.enhance_modules[stage_idx](xR, xL)

            # 应用融合模块（若当前阶段需要融合）
            if stage_name in self.fusion_modules:
                x_fusion = self.fusion_modules[stage_name](xR_prime, xL_prime)
                outputs[stage_name] = x_fusion

            # 传递增强后的特征到下一阶段
            xR, xL = xR_prime, xL_prime

        # 分类头逻辑（若有）
        if self.num_classes is not None:
            x = F.adaptive_avg_pool2d(outputs["res5"], (1, 1))
            x = torch.flatten(x, 1)
            x = self.linear(x)
            outputs["linear"] = x

        return {k: v for k, v in outputs.items() if k in self._out_features}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            ) for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        self.branch_R.freeze(freeze_at)
        self.branch_L.freeze(freeze_at)
        return self


