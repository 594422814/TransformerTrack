import os
import sys
import argparse

import torch
import pdb

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


pretrain_net = torch.load('/home/wangning/Tracking/pytracking/workspace/checkpoints/ltr/dimp/my_super_dimp_mask/DiMPnet_ep0050.pth.tar') 
pdb.set_trace()
res = pretrain_net['stats']['train']['Loss/target_clf'].history
# pretrain_net['stats']['train']['Loss/test_init_clf'].history

if __name__ == '__main__':
    main()
