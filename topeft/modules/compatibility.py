import numpy as np


def add_sumw2_stub(eval_d):
    # compatibility with the coffea.Hist. add a row of zeros for sumw2.
    eval_d2 = {}
    for k, v in eval_d.items():
        eval_d2[k] = np.stack((v, np.broadcast_to(np.zeros((1,)), len(v))))
    return eval_d2
