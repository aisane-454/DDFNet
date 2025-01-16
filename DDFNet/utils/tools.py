import os
import torch
import random
import numpy as np

def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

SEED = 3407
def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable =True
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    global SEED
    random.seed(SEED+worker_id)
