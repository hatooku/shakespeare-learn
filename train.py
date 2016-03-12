# OFF THE SHELF IMPLEMENTATION USE

from hmm_helper import *
from hmm import HMM as myHMM
import numpy as np
from hmmlearn import hmm
import time

def processData(data):
    D = 0
    token_to_num = {}

    X = None # data transformed into integers corresponding to tokens
    length_arr = []
    for seq in data:
        X_i = [] # this sequence transformed into integers
        for token in seq:
            if token not in token_to_num:
                token_to_num[token] = D
                D += 1
            X_i.append([token_to_num[token]])


        if X is None:
            X = np.array(X_i)
        else:
            X = np.concatenate([X, np.array(X_i)])
        length_arr.append(len(X_i))

    token_dict = {v: k for k, v in token_to_num.items()}

    return (X, length_arr, token_to_num)


NUM_STATES = 30

# Process Data to be used with hmm
data = shksp_quatrain_couplets_line(simple_token2,
                                    filename='shakespeare.txt')
data_quatrain = data[0]
data_couplet = data[1]

X_q, length_arr_q, token_to_num_q = processData(data_quatrain)
X_c, length_arr_c, token_to_num_c = processData(data_couplet)

# QUATRAIN
start_time = time.time()
model_q = hmm.MultinomialHMM(n_components=NUM_STATES,  n_iter=100)
model_q.fit(X_q, lengths=length_arr_q)

print "time:", (time.time() - start_time)

PI_q = model_q.startprob_
A_q = model_q.transmat_
O_q = model_q.emissionprob_

save_model("Joon_30_Token2_Quatrain", A_q, O_q, token_to_num_q, PI_q)

# COUPLET
start_time = time.time()
model_c = hmm.MultinomialHMM(n_components=NUM_STATES,  n_iter=100)
model_c.fit(X_c, lengths=length_arr_c)

print "time:", (time.time() - start_time)

PI_c = model_c.startprob_
A_c = model_c.transmat_
O_c = model_c.emissionprob_

save_model("Joon_30_Token2_Couplet", A_c, O_c, token_to_num_c, PI_c)

