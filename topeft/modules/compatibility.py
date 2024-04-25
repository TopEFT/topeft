import numpy as np


def add_sumw2_stub(eval_d, sumw2=False):
    # compatibility with the coffea.Hist. add a row of zeros for sumw2.
    eval_d2 = {}
    for k, v in eval_d.items():
        if not sumw2:
            eval_d2[k] = np.stack((v, np.sqrt(sumw2[k])))
        else:
            eval_d2[k] = np.stack((v, np.broadcast_to(np.zeros((1,)), len(v))))
    return eval_d2
