import pdb
import sys
import torch

from utils import basic_utils

from eval import *
from model import *
from dataloader import *
from utils.scheduler import *
from cfgs.options import *
from cfgs.config import *


def read_txt(path, type=None):
    txt_list = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip('\n')
            if type == 'int+':
                txt_list.append(int(line)+1)
            elif type == 'int':
                txt_list.append(int(line))
            else:
                txt_list.append(line)
    return txt_list


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config_egd_lsfit(args)

    if config.seed >= 0:
        basic_utils.set_seed(config.seed)
    model = Model_VAC(config)
    model = model.cuda()

    state_dict = torch.load(args.model_file)
    model.load_state_dict(state_dict)

    save_dir='video_pred_online'
    test_list = read_txt('./dataset/EGD/test_list.txt',type='str')
    for test_path in test_list:
        patient_id = os.path.dirname(test_path[:-1])
        print(patient_id)
        eval_VAC(patient_id, config, model, dir=save_dir)
