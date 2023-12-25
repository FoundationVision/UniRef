# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_uniref_config(cfg):
    """
    Add config for UniRef.
    """
    # general settings
    cfg.MODEL.OTA = False
    cfg.MODEL.TEST_NMS = 0.7
    cfg.MODEL.WITH_LANG_REF = False     # whether use text encoder
    cfg.MODEL.WITH_MASK_REF = False     # whether use mask encoder
    cfg.MODEL.USE_EARLY_FUSION = True   # whether perform early-fusion in Transformer encoder.
    cfg.MODEL.MERGE_ON_CPU = True       # for video inference

    # Unified dataloader for multiple tasks
    # cfg.DATALOADER.SAMPLER_TRAIN = "MultiDatasetSampler"
    cfg.DATALOADER.DATASET_RATIO = [1, 1]
    cfg.DATALOADER.USE_DIFF_BS_SIZE = True
    cfg.DATALOADER.DATASET_BS = [2, 2]
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.DATASET_ANN = ['box', 'image']

    # Allow different datasets to use different input resolutions
    cfg.INPUT.MIN_SIZE_TRAIN_MULTI = [(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), (320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640)]
    cfg.INPUT.MAX_SIZE_TRAIN_MULTI = [1333, 768]

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_FRAME_RANGE_VOS = None   # if specify, then use this value for vos
    cfg.INPUT.SAMPLING_FRAME_RANGE_RVOS = None  # if specify, then use this value for rvos
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    #  LANGUAGE SETTINGS
    cfg.MODEL.LANG_CONFIG = CN()
    cfg.MODEL.LANG_CONFIG.FREEZE_TEXT_ENCODER = True
    cfg.MODEL.LANG_CONFIG.MODEL_TYPE = "bert-base"
    cfg.MODEL.LANG_CONFIG.LANG_DIM = 768      # roberta dim
    cfg.MODEL.LANG_CONFIG.CONTEXT_LEN = 77    # the same as CLIP
    cfg.MODEL.LANG_CONFIG.LANG_POOL = True    # If True, pool the word features; else, use [cls] of Text Encoder.

    # TASK SETTING
    cfg.TASK = CN()
    cfg.TASK.FSS = CN()
    cfg.TASK.FSS.NSHOT = 1

    # BoxInst
    cfg.MODEL.BOXINST = CN()
    cfg.MODEL.BOXINST.ENABLED = False # Whether to enable BoxInst
    cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10
    cfg.MODEL.BOXINST.PAIRWISE = CN()
    cfg.MODEL.BOXINST.PAIRWISE.SIZE = 3
    cfg.MODEL.BOXINST.PAIRWISE.DILATION = 2
    cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
    cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3
    cfg.MODEL.BOXINST.TOPK = 64 # max number of proposals for computing mask loss

    # DataLoader
    cfg.INPUT.DATASET_MAPPER_NAME = "detr" # use "coco_instance_lsj" for LSJ aug
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    # Larger input size
    cfg.INPUT.IMAGE_SIZE_LARGE = 1024 # 1536
    # mixup
    cfg.INPUT.USE_MIXUP = False
    cfg.INPUT.MIXUP_PROB = 1.0

    # Transformer Config
    cfg.MODEL.DDETRS = CN()
    cfg.MODEL.DDETRS.NUM_CLASSES = 80
    cfg.MODEL.DDETRS.TO_AGNOSTIC = False
    cfg.MODEL.DDETRS.USE_CHECKPOINT = False # whether to use gradient-checkpoint for the transformer

    # LOSS
    cfg.MODEL.DDETRS.MASK_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DICE_WEIGHT = 5.0
    cfg.MODEL.DDETRS.GIOU_WEIGHT = 2.0
    cfg.MODEL.DDETRS.L1_WEIGHT = 5.0
    cfg.MODEL.DDETRS.CLASS_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DEEP_SUPERVISION = True
    cfg.MODEL.DDETRS.MASK_STRIDE = 4
    cfg.MODEL.DDETRS.MASK_STRIDE_VOS = 1     # VOS supervised in the original resolution
    cfg.MODEL.DDETRS.MATCH_STRIDE = 4
    cfg.MODEL.DDETRS.FOCAL_ALPHA = 0.25

    cfg.MODEL.DDETRS.MASK_AUX_LOSS = False   # whether use additional MaskIOU loss.
    cfg.MODEL.DDETRS.MASK_AUX_WEIGHT = 1.0

    cfg.MODEL.DDETRS.SET_COST_CLASS = 2
    cfg.MODEL.DDETRS.SET_COST_BOX = 5
    cfg.MODEL.DDETRS.SET_COST_GIOU = 2

    # TRANSFORMER
    cfg.MODEL.DDETRS.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.DDETRS.NHEADS = 8
    cfg.MODEL.DDETRS.DROPOUT = 0.1
    cfg.MODEL.DDETRS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DDETRS.ENC_LAYERS = 6
    cfg.MODEL.DDETRS.DEC_LAYERS = 6
    cfg.MODEL.DDETRS.TWO_STAGE = False
    cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS = 300
    cfg.MODEL.DDETRS.MIXED_SELECTION = False
    cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE = False
    cfg.MODEL.DDETRS.USE_IOU_BRANCH = False

    cfg.MODEL.DDETRS.HIDDEN_DIM = 256
    cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.DDETRS.DEC_N_POINTS = 4
    cfg.MODEL.DDETRS.ENC_N_POINTS = 4
    cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS = 4

    # Mask Postprocessing & Upsampling
    cfg.MODEL.DDETRS.MASK_THRES = 0.5
    cfg.MODEL.DDETRS.NEW_MASK_HEAD = False
    cfg.MODEL.DDETRS.USE_RAFT = False
    cfg.MODEL.DDETRS.USE_REL_COORD = True
    cfg.MODEL.DDETRS.USE_BACKBONE_FEATURE = False  # whether use the backbone 4x feature for mask dynamic convolution

    # DINO, Denoising， TODO: delete this
    cfg.MODEL.DDETRS.USE_DINO = False
    cfg.MODEL.DDETRS.DN_NUMBER = 100
    cfg.MODEL.DDETRS.LABEL_NOISE_RATIO = 0.5
    cfg.MODEL.DDETRS.BOX_NOISE_SCALE = 1.0

    # Early-fusion 
    cfg.MODEL.DDETRS.FUSE_CONFIG = CN()
    cfg.MODEL.DDETRS.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
    cfg.MODEL.DDETRS.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
    cfg.MODEL.DDETRS.FUSE_CONFIG.STABLE_SOFTMAX_2D = False
    cfg.MODEL.DDETRS.FUSE_CONFIG.HEAD_DIM = 256
    cfg.MODEL.DDETRS.FUSE_CONFIG.NHEADS = 8

    # Optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.LINEAR_PROJ_MULTIPLIER = 0.1
    cfg.SOLVER.LR_TEXT_ENCODER = 1e-5
    cfg.SOLVER.LR_DECAY_RATE = None  # 0.8
    cfg.SOLVER.LR_DECAY_RATE_NUM_LAYERS = 24

    
    # BACKBONE
    # ResNet
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]   
    # Swin
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.NAME = "tiny"
    cfg.MODEL.SWIN.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    # ConvNext
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.NAME = "tiny"
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False
    # supprt ViT backbone
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.NAME = "ViT-Base"
    cfg.MODEL.VIT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.VIT.USE_CHECKPOINT = False
    cfg.MODEL.VIT.POS_EMB_FORM = "leanable"
    # EVA02
    cfg.MODEL.EVA02 = CN()
    cfg.MODEL.EVA02.NAME = "large"
    cfg.MODEL.EVA02.OUT_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.EVA02.USE_TOP_BLOCK = True
    cfg.MODEL.EVA02.USE_CHECKPOINT = False

    # find_unused_parameters, TODO: fix unused in code
    cfg.FIND_UNUSED_PARAMETERS = True

    # freeze detr detector
    cfg.FREEZE_DETR = False
