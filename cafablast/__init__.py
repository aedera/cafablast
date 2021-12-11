import os

import numpy as np

import scipy.sparse
from scipy.sparse import csr_matrix

_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(_dir, 'data')

cc_scores_path = os.path.join(data_dir, 'cc_scores.npz')
bp_scores_path = os.path.join(data_dir, 'bp_scores.npz')
mf_scores_path = os.path.join(data_dir, 'mf_scores.npz')

cc_gterms_path = os.path.join(data_dir, 'cc_gterms.npz')
bp_gterms_path = os.path.join(data_dir, 'bp_gterms.npz')
mf_gterms_path = os.path.join(data_dir, 'mf_gterms.npz')

def load_scores(scores_path):
    raw = np.load(scores_path, allow_pickle=True)

    mat = csr_matrix((raw['val'], (raw['row'], raw['col'])), shape=raw['shape'])

    return mat, raw['p2r'].item()

def load_gterms(gterms_path):
    raw = np.load(gterms_path, allow_pickle=True)

    mat = csr_matrix((raw['val'], (raw['row'], raw['col'])), shape=raw['shape'])

    c2t = {c:t for t, c in raw['t2c'].item().items()}

    return mat, c2t


# load matrices and mappings
CC_SCR, CC_P2R = load_scores(cc_scores_path)
BP_SCR, BP_P2R = load_scores(bp_scores_path)
MF_SCR, MF_P2R = load_scores(mf_scores_path)

CC_TRM, CC_C2T = load_gterms(cc_gterms_path)
BP_TRM, BP_C2T = load_gterms(bp_gterms_path)
MF_TRM, MF_C2T = load_gterms(mf_gterms_path)

# print(CC_SCR.shape)
# print(CC_TRM.shape)

# print(BP_SCR.shape)
# print(BP_TRM.shape)

# print(MF_SCR.shape)
# print(MF_TRM.shape)

CC = CC_SCR @ CC_TRM
BP = BP_SCR @ BP_TRM
MF = MF_SCR @ MF_TRM

#breakpoint()
del CC_TRM, BP_TRM, MF_TRM

from .main import predict
