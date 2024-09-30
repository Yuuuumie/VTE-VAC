import collections
import time
import torch
import torch.nn as nn
import numpy as np
import os
from dataloader import *
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)

text_features = np.load("./dataset/EGD/Text-features-bert/texts-feature.npy")
vocab_size = text_features.shape[0]
text_dict = {}
for i in range(vocab_size):
    text_dict[i] = torch.tensor(text_features[i:i+1,:])

def get_savename(patient_name):
    if patient_name.find('qualify')>-1:
        patient_name = patient_name.split('qualify/')[-1]
    savename = patient_name.replace('/','_')
    return savename


def fliter(predict_label):
    num = 0
    for i in range(len(predict_label)):
        if predict_label[i] == 1:
            num = num + 1
        else:
            if num <= 5:
                predict_label[i - num:i] = [0] * num
            num = 0
        if i == len(predict_label) - 1:
            if num <= 5:
                predict_label[i - num+1:i+1] = [0] * num
            num = 0
    return predict_label


def convert_id_to_features(cls_ids):
        text_features = np.concatenate([text_dict[id] for id in cls_ids], axis=0)
        return text_features


def to_device(x, dtype=np.float32):
        return torch.as_tensor(x.astype(dtype)).unsqueeze(0).cuda()


def postprocessing(roi, site):
    roi = fliter(roi)
    origin_preds = np.multiply(site+1,roi)-1
    origin_preds[origin_preds == -1] = 54
    processing_preds = []
    all = 0
    for i in range(len(roi)):
        if roi[i] == 0:
            processing_preds.append(54)
            if all > 0:
                pd = origin_preds[i-all:i]
                num = collections.Counter(pd)
                data = num.most_common(1)
                label = data[0][0]
                label_num = data[0][1]
                processing_preds[i-all:i]=[label]*all
                all=0
        else:
            processing_preds.append(origin_preds[i])
            all = all + 1
    return origin_preds, processing_preds


def eval_VAC(patient_id, cfg, model, dir):
    bs = 1
    assert (bs == 1), 'batch_size must be 1 for model lsfit stream inference '
    model.eval()

    test_dataset = EGDFeature_onlinetest(cfg,patient_id)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=bs,
        num_workers=cfg.num_workers,
        pin_memory=False,
        )
    
    # Collect scores and targets
    site_preds = []
    site_confs = []
    roi_preds = []
    temp_preds = []
    # temp_site_counts = 0
    work_memory_num_samples = cfg.work_memory_length//cfg.work_memory_sampling_rate

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            img_feature, memory_key_padding_mask, _, _, work_indices, long_indices, _ = data

            work_start = work_indices[0][0]
            work_end = work_indices[0][-1]
            long_indices = long_indices.data.cpu().numpy().tolist()[0]

            # inital text work memory
            if work_start == 0:
                work_text_anno_cache = [54]*work_memory_num_samples
                work_text_anno_cache[-1] = 55
            work_text_feature = to_device(convert_id_to_features(work_text_anno_cache))

            # inital text long memory 
            long_end = work_start - 1
            if long_end == -1:
                long_text_anno_cache = [54]*cfg.long_memory_num_samples
            long_text_feature = to_device(convert_id_to_features(long_text_anno_cache))

            img_feature = img_feature.cuda()
            work_text_feature = work_text_feature.cuda()
            long_text_feature = long_text_feature.cuda()
            memory_key_padding_mask = memory_key_padding_mask.cuda()
            if cfg.work_mask_ratio == 1.:
                text_feature = long_text_feature
            else:
                text_feature = torch.cat((long_text_feature, work_text_feature),dim=1)

            site_score, roi_score = model(img_feature, text_feature, memory_key_padding_mask)

            site_score, roi_score = site_score.squeeze(0).cpu().numpy(), roi_score.squeeze(0).cpu().numpy()

            # site predict class
            site_pred = np.argmax(site_score,axis=1)
            site_conf = site_score

            # roi predict class
            background_score = roi_score[:,0]
            roi_pred = 1-background_score
            roi_pred[roi_pred>=0.5] = 1
            roi_pred[roi_pred<0.5] = 0
            
            # temp predict class for text memory 
            temp_pred = np.multiply(site_pred+1,roi_pred)-1
            temp_pred[temp_pred == -1] = 54

            if work_start == 0:
                site_preds.extend(list(site_pred))
                site_confs.extend(list(site_conf))
                roi_preds.extend(list(roi_pred))
                temp_preds.extend(list(temp_pred))
            else:
                site_preds.append(list(site_pred)[-1])
                site_confs.append(list(site_conf)[-1])
                roi_preds.append(list(roi_pred)[-1])
                temp_preds.append(list(temp_pred)[-1])

            # update long text memory cache
            long_text_anno_cache = np.array(temp_preds)[long_indices]
            # update work text memory cache
            work_text_anno_pred = temp_preds[-1]
            work_text_anno_cache = work_text_anno_cache[1:work_memory_num_samples-1]+[work_text_anno_pred]+[55]
   
    site_confs = np.asarray(site_confs)
    site_preds = np.asarray(site_preds)
    roi_preds = np.asarray(roi_preds)
    temp_preds = np.asarray(temp_preds)

    # postprocess
    final_preds, pred_segmax = postprocessing(roi_preds, site_preds)

    save_path = os.path.join(cfg.output_path,f'eval/{dir}/')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = get_savename(patient_id)
    
    conf_path = save_path+save_name+'_conf.txt'
    np.savetxt(conf_path,site_confs,fmt = '%6.4f')

    pred_path = save_path+save_name+'_pred.txt'
    np.savetxt(pred_path,final_preds,fmt = '%d')

    pred_segmax_path = save_path+save_name+'_pred_segmax.txt'
    np.savetxt(pred_segmax_path,pred_segmax,fmt='%d')
