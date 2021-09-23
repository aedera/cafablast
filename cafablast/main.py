import numpy as np

import scipy
import scipy.sparse
from scipy.sparse import csr_matrix

from . import CC, BP, MF
from . import CC_P2R, BP_P2R, MF_P2R
from . import CC_C2T, BP_C2T, MF_C2T

def predict(query, namespace='cc'):
    if namespace == 'cc':
        map_p2r = CC_P2R
        mat     = CC
        map_c2t = CC_C2T
    elif namespace == 'bp':
        map_p2r = BP_P2R
        mat     = BP
        map_c2t = BP_C2T
    elif namespace == 'mf':
        map_p2r = MF_P2R
        mat     = MF
        map_c2t = MF_C2T
    else:
        print("Unknown namespace")
        exit(0)

    i = map_p2r[query] # retrieve row number
    row_i = mat.getrow(i).todense() # get bitscores for terms
    probs = row_i / np.sum(row_i) # normalize

    predicted_terms = {}
    # loop over terms with non-zero probability
    for k in np.where(probs > 0)[1]:
        predicted_terms[map_c2t[k]] = probs[0,k]

    return predicted_terms
