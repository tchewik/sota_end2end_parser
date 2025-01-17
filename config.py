# -*- coding: utf-8 -*-
from path_config import *
from util.file_util import *


SAVE_MODEL = True
VERSION, SET = 0, 2
SEED = 19
USE_CUDA, CUDA_ID = False, 'cpu'
EDU_ENCODE_VERSION, SPLIT_V = 2, 1
CHUNK_SIZE = 768  # 64
MAX_LEN = 32
USE_BOUND, USE_GAN, USE_S_STACK = True, True, False
TRAIN_XLNET, XL_FINE, Joint_EDU_R, XLNET_TYPE, XLNET_SIZE = True, True, False, "xlnet-base-cased", 768
MAX_W, MAX_H = 80, 20
LEARN_RATE, WD, D_LR = 0.0001, 1e-4, 0.0001
EPOCH, BATCH_SIZE, LOG_EVERY, VALIDATE_EVERY = 60, 1, 5, 20
UPDATE_ITE = 32
WARM_UP_EP = 7 if USE_GAN else EPOCH
in_channel_G, out_channel_G, ker_h_G, ker_w_G, strip_G = 2, 32, 3, MAX_W // 2, 1
p_w_G, p_h_G = 3, 3
METype = 1
USE_AE = False
GATE_V, GateDrop, OPT_ATTN = 1, 0.2, False
MAX_POOLING = True
RANDOM_MASK_LEARN, RMR = False, 0.1
USE_POSE, POS_SIZE = False, 30
USE_ELMo, EMBED_LEARN = False, False
EMBED_SIZE, HIDDEN_SIZE = (1024 if USE_ELMo else 300), 384
BOUND_INFO_SIZE = 30  # 30
EDU_ATT, ML_ATT_HIDDEN_e, HEADS_e = False, 128, 2
CONTEXT_ATT, ML_ATT_HIDDEN, HEADS = True, 128, 2
SPLIT_MLP_SIZE, NR_MLP_SIZE = 128, 128
USE_CNN, KERNEL_SIZE, PADDING_SIZE = True, 2, 1
LAYER_NORM_USE = True
ALPHA_SPAN, ALPHA_NR = 0.3, 1.0
L2, DROP_OUT = 1e-5, 0.2
TRAN_LABEL_NUM, NR_LABEL_NUM, NUCL_LABEL_NUM, REL_LABEL_NUM = 1, 42, 3, 18
SHIFT, REDUCE = "SHIFT", "REDUCE"
REDUCE_NN, REDUCE_NS, REDUCE_SN = "REDUCE-NN", "REDUCE-NS", "REDUCE-SN"
NN, NS, SN = "NN", "NS", "SN"
PAD, PAD_ids = "<PAD>", 0
UNK, UNK_ids = "<UNK>", 1
action2ids = {SHIFT: 0, REDUCE: 1}
ids2action = {0: SHIFT, 1: REDUCE}
nucl2ids = {NN: 0, NS: 1, SN: 2} if METype == 1 else {"N": 0, "S": 1}
ids2nucl = {0: NN, 1: NS, 2: SN} if METype == 1 else {0: "N", 1: "S"}
ns_dict = {"Satellite": 0, "Nucleus": 1, "Root": 2}
ns_dict_ = {0: "Satellite", 1: "Nucleus", 2: "Root"}
coarse2ids = load_data(REL_coarse2ids)
ids2coarse = load_data(REL_ids2coarse)
nr2ids = load_data(LABEL2IDS)
ids2nr = load_data(IDS2LABEL)