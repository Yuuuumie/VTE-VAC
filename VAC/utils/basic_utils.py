import torch
import random
import numpy as np
from scipy.interpolate import interp1d

torch.set_printoptions(profile="full")

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record_egd(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.close()


def save_best_record_sfb(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("threshold: {:.4f}\n".format(test_info["threshold"][-1]))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.write("Test_recall: {:.4f}\n".format(test_info["test_recall"][-1]))
    fo.close()


def save_best_record_edit(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Test_Dice: {:.4f}\n".format(test_info["test_Dice"][-1]))
    fo.write("Test_Edit: {:.4f}\n".format(test_info["test_Edit"][-1]))
    fo.write("Test_F1@0.1: {:.4f}\n".format(test_info["test_F1@0.1"][-1]))
    fo.write("Test_F1@0.25: {:.4f}\n".format(test_info["test_F1@0.25"][-1]))
    fo.write("Test_F1@0.5: {:.4f}\n".format(test_info["test_F1@0.5"][-1]))
    fo.close()


def save_best_record_mAP(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("mAP: {:.4f}\n".format(test_info["mAP"][-1]))
    fo.close()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()
