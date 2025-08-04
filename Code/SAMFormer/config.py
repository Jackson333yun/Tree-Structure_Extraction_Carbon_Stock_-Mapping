# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper

    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    #其它新增参数
    cfg.MODEL.BACKBONE.FUSE_STAGES = [3, 4, 5] #如果使用AdaoEnhenCroModalFusnet，指定在哪个stage进行融合
    cfg.MODEL.BACKBONE.ENHANCE_MODULE = "None_Enhence"  # 使用通道注意力
    cfg.MODEL.BACKBONE.FUSION_MODULE = "ConcatFusion"  # 使用简单融合模块
    cfg.MODEL.BACKBONE.ALTERNATE_FUSION_MODULE = "ConcatFusion"    #未指定在哪个阶段进行融合的剩余部分，使用何种融合方式
    cfg.MODEL.BACKBONE.ATTENTION_MODULE = "StandardCrossAttention"

    cfg.MODEL.CLASS_FREQUENCIES = [2156, 602, 1183, 59, 267, 171]
    cfg.MODEL.CB_LOSS_BETA = 0.9999
    cfg.SOLVER.SKIP_PERIOD = 500


"""
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN  # 引入 ConfigNode 用于配置管理


def add_maskformer2_config(cfg):
    # NOTE: configs from original maskformer

    # 数据配置部分
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"  # 数据集映射器名称，指定为实例分割任务的映射器
    cfg.INPUT.COLOR_AUG_SSD = False  # 是否启用颜色增强，设置为 False 表示禁用
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0  # 随机裁剪时，确保任何类别占据的区域不超过最大区域，设置为 1.0 表示没有限制
    cfg.INPUT.SIZE_DIVISIBILITY = -1  # 设置为 -1 表示不对图像尺寸进行特殊要求（大小可不被整除）

    # 优化器和学习率配置
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0  # 对 embedding 层的权重衰减，设置为 0 表示不进行衰减
    cfg.SOLVER.OPTIMIZER = "ADAMW"  # 使用 AdamW 优化器
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1  # backbone 学习率的倍率，通常 backbone 学习率较低，设置为 0.1

    # MaskFormer 模型的配置
    cfg.MODEL.MASK_FORMER = CN()  # 创建 MASK_FORMER 配置项

    # 损失函数相关配置
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True  # 启用深层监督
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1  # 缺失物体的权重
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0  # 类别损失的权重
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0  # Dice 损失的权重
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0  # 掩码损失的权重

    # Transformer 配置
    cfg.MODEL.MASK_FORMER.NHEADS = 8  # Transformer 中 attention 头的数量
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1  # dropout 比率
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048  # 前馈神经网络的维度
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0  # Transformer 编码器层数，设置为 0 表示不使用编码器
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6  # Transformer 解码器层数
    cfg.MODEL.MASK_FORMER.PRE_NORM = False  # 是否在每层前进行归一化，设置为 False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256  # Transformer 中隐藏层的维度
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100  # 每个解码器的对象查询数

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"  # Transformer 输入的特征层，这里是 res5
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False  # 是否强制进行输入特征的投影，设置为 False

    # MaskFormer 推理时的配置
    cfg.MODEL.MASK_FORMER.TEST = CN()  # 创建推理配置项
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True  # 启用语义分割推理
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False  # 禁用实例分割推理
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False  # 禁用全景分割推理
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0  # 对象掩码的阈值，设置为 0 表示没有阈值
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0  # 重叠阈值，设置为 0 表示没有阈值
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False  # 在推理前是否进行语义分割后处理

    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32  # 设置模型输入的尺寸必须是 32 的倍数（用于某些 backbone）

    # Pixel Decoder 配置
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256  # 掩码的维度
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0  # Pixel Decoder 中 Transformer 编码器层数
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"  # Pixel Decoder 的名称

    # Swin Transformer Backbone 配置
    cfg.MODEL.SWIN = CN()  # 创建 Swin Transformer 配置项
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224  # Swin Transformer 的预训练图像大小
    cfg.MODEL.SWIN.PATCH_SIZE = 4  # Swin Transformer 中 patch 的大小
    cfg.MODEL.SWIN.EMBED_DIM = 96  # 嵌入的维度
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]  # Swin Transformer 的每层深度
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]  # 每层的 multi-head attention 头数
    cfg.MODEL.SWIN.WINDOW_SIZE = 7  # 窗口的大小
    cfg.MODEL.SWIN.MLP_RATIO = 4.0  # MLP 层的比例
    cfg.MODEL.SWIN.QKV_BIAS = True  # 是否启用 QKV 的偏置
    cfg.MODEL.SWIN.QK_SCALE = None  # QK 的缩放系数，设置为 None 表示不使用缩放
    cfg.MODEL.SWIN.DROP_RATE = 0.0  # dropout 比率
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0  # attention dropout 比率
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3  # drop path 的比率
    cfg.MODEL.SWIN.APE = False  # 是否启用位置编码
    cfg.MODEL.SWIN.PATCH_NORM = True  # 是否在每个 patch 上进行归一化
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]  # Swin 输出的特征层
    cfg.MODEL.SWIN.USE_CHECKPOINT = False  # 是否使用 checkpoint

    # MaskFormer2 的额外配置项
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"  # 使用多尺度的 Masked Transformer 解码器

    # LSJ (Large Scale Jitter) 数据增强配置
    cfg.INPUT.IMAGE_SIZE = 1024  # 输入图像的尺寸
    cfg.INPUT.MIN_SCALE = 0.1  # 最小缩放比例
    cfg.INPUT.MAX_SCALE = 2.0  # 最大缩放比例

    # MSDeformAttn 编码器配置
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]  # MSDeformAttn 编码器的输入特征层
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4  # MSDeformAttn 编码器的每个点的采样数
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8  # MSDeformAttn 编码器的头数

    # 点损失配置
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112  # 训练时，每个掩码点头部采样的点数
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0  # PointRend 点采样的过采样比率
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75  # PointRend 重要性采样的比率

"""