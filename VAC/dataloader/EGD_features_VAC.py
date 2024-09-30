import copy
import math
import os
import random
from bisect import bisect_right

import numpy as np
import torch
import torch.utils.data as data

from utils import basic_utils


def _get_action_dict(mapping_path):
    file_ptr = open(mapping_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[int(a.split()[0])] = a.split()[1]
    return actions_dict

def read_txt(self, path, type=None):
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


class EGDFeature_train(data.Dataset):
    def __init__(self, config):
        if config.seed >= 0:
            basic_utils.set_seed(config.seed)

        self.modal = config.modal
        self.data_path = config.data_path
        self.work_memory_length = config.work_memory_length
        self.long_memory_length = config.long_memory_length
        self.long_memory_sampling_rate = config.long_memory_sampling_rate
        self.long_memory_num_samples = self.long_memory_length//self.long_memory_sampling_rate
        
        self.train_list = read_txt('./dataset/EGD/train_list.txt',type='str')
        self.action_dict = _get_action_dict('./dataset/EGD/mapping.txt')

        self.text_dict = {}
        text_features = np.load("./dataset/EGD/Text-features-bert/texts-feature.npy")
        vocab_size,self.d_text = text_features.shape
        for i in range(vocab_size):
            self.text_dict[i] = torch.tensor(text_features[i:i+1,:])
        self.classes = {
            0:[1,2], 1:[0,2], 2:[0,1], 

            3:[4,5,6,7], 4:[3,5,6,7], 5:[3,4,6,7], 6:[3,4,5,7], 7:[3,4,5,6],

            8:[9,10,38,42,46],9:[8,10,38,42,46],10:[8,9,38,42,46],

            11:[12,13,14,15,16],12:[11,13,14,15,16],13:[11,12,14,15,16],
            14:[11,12,13,15,16],15:[11,12,13,14,16],16:[11,12,13,14,15],

            17:[18],18:[17],

            19:[20,21],20:[19,21],21:[19,20],

            22:[23],23:[27,30,33],24:[28,31,34],25:[29,32,35],

            26:[22,23],27:[30,33],28:[31,34],
            29:[32,35],30:[27,33],31:[28,34],

            32:[29,35],33:[27,30],34:[28,31],35:[29,32],
            36:[40,44],37:[41,45],38:[42,46],39:[43,47],
            40:[36,44],41:[37,45],42:[38,46],43:[39,47],
            44:[36,40],45:[37,41],46:[38,42],47:[39,43],

            48:[49,50,51,52,53],49:[48,50,51,52,53],50:[48,49,51,52,53],
            51:[48,49,50,52,53],52:[48,49,50,51,53],53:[48,49,50,51,52]}
        
        self._init_dataset()

    def convert_id_to_features(self, cls_ids):
        text_features = torch.cat([self.text_dict[id] for id in cls_ids], dim=0)
        return text_features
    
    def mask_replace_memory(self, memory, text_anno):
        if random.random() < 0.8:
            # mask 75% label
            mask_index = random.sample(range(0,memory.size(0)-1),int(memory.size(0)*0.75))
            masked_memory = torch.ones(memory.size(0),memory.size(1))
            masked_memory[mask_index,:] = 0
            masked_memory_output = torch.multiply(memory,masked_memory)
            return masked_memory_output
        else:
            if random.random() < 0.5:
                # 不变
                return memory
            else:
                # replace 50% cls
                cls_pool = list(set(text_anno))
                if 54 in cls_pool:
                    cls_pool.remove(54)
                if 55 in cls_pool:
                    cls_pool.remove(55)
                cls_replace_pool = random.sample(cls_pool,math.ceil(len(cls_pool)/2))
                replaced_text_anno = copy.deepcopy(text_anno)
                for cls in cls_replace_pool:
                    replace_index = np.where(text_anno == cls)[0]
                    cls_replace = random.choice(self.classes[cls])
                    replaced_text_anno[replace_index] = cls_replace
                replaced_memory_output = self.convert_id_to_features(replaced_text_anno)
                return replaced_memory_output




    def shuffle(self):
        self._init_dataset()


    def _init_dataset(self):
        self.inputs = []
        for train_path in self.train_list:

            train_path = os.path.dirname(train_path[:-1])
            patient_id = train_path.split('Video/')[-1]
            target_path = self.get_target_path(patient_id)
            target = np.array(read_txt(target_path,type='int'))
            seed = np.random.randint(self.work_memory_length)
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.work_memory_length),
                range(seed+self.work_memory_length, target.shape[0], self.work_memory_length)):

                work_anno = target[work_start:work_end]
                if not (work_anno == np.ones_like(work_anno)*54).all():
                    self.inputs.append([patient_id, work_start, work_end, work_anno, target])
                    
        print(len(self.inputs))

    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        patient_id, work_start, work_end, work_anno, target = self.inputs[index]

        # Get img feature
        if self.modal == 'fusion':
            site_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            roi_feature_path = os.path.join(self.data_path.replace('-cotnet','-SFB-cotnet'),patient_id,'features.npy')
            site_feature = np.load(site_feature_path, mmap_mode='r')
            roi_feature = np.load(roi_feature_path, mmap_mode='r')
            img_feature = np.hstack((site_feature,roi_feature))
        elif self.modal in ['multitask','rgb','roi','I3D']:
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        else:
            raise(AssertionError)
        
        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        # Get img work memory
        work_img_feature = img_feature[work_indices]
        # Get text work memory
        work_text_anno = copy.deepcopy(work_anno)
        work_text_anno[-1] = 55

        work_text_feature = self.convert_id_to_features(work_text_anno)
        work_text_feature = self.mask_replace_memory(work_text_feature, work_text_anno)
        
        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.segment_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples).clip(0)
                
            # Get img long memory
            long_img_feature = img_feature[long_indices]
            # Get text long memory
            long_text_anno = target[long_indices]
            long_text_feature = self.convert_id_to_features(long_text_anno)
            long_text_feature= self.mask_replace_memory(long_text_feature, long_text_anno)
            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_img_feature = None
            long_text_feature = None
            memory_key_padding_mask = None
        
        # Get all memory
        if long_img_feature is not None and work_text_feature is not None:
            input_img_feature = np.concatenate((long_img_feature, work_img_feature))
            input_text_feature = np.concatenate((long_text_feature, work_text_feature))
        elif long_img_feature is not None and work_text_feature is None:
            input_img_feature = np.concatenate((long_img_feature, work_img_feature))
            input_text_feature = np.array(long_text_feature)
        else:
            input_img_feature = work_img_feature
            input_text_feature = work_text_feature

        # Convert to tensor
        input_img_feature = torch.as_tensor(input_img_feature.astype(np.float32))
        input_text_feature = torch.as_tensor(input_text_feature.astype(np.float32))
        anno = torch.as_tensor(work_anno)
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return input_img_feature, input_text_feature, memory_key_padding_mask, anno
        else:
            return input_img_feature, input_text_feature, anno


    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)


    def get_target_path(self,patient_name):
        gt_path = './dataset/EGD/foreground_gt/txt'
        element = patient_name.split('/')
        if len(element)>2:
            date_id = element[0]
            gtname = date_id+patient_name.split(element[1])[-1].replace('/','_')
        else:
            gtname = patient_name.replace('/','_')
        return os.path.join(gt_path,gtname+'.txt')
    

class EGDFeature_batchtest(data.Dataset):
    def __init__(self, config, patient_id):
        if config.seed >= 0:
            basic_utils.set_seed(config.seed)

        self.modal = config.modal
        self.data_path = config.data_path
        self.work_memory_length = config.work_memory_length
        self.long_memory_length = config.long_memory_length
        self.long_memory_sampling_rate = config.long_memory_sampling_rate
        self.long_memory_num_samples = self.long_memory_length//self.long_memory_sampling_rate

        self.inputs = []

        self.action_dict = _get_action_dict('./dataset/EGD/mapping.txt')

        self.text_dict = {}
        text_features = np.load("./dataset/EGD/Text-features-bert/texts-feature.npy")
        vocab_size,self.d_text = text_features.shape
        for i in range(vocab_size):
            self.text_dict[i] = torch.tensor(text_features[i:i+1,:])

        target_path = self.get_target_path(patient_id)
        target = np.array(read_txt(target_path,type='int'))

        for work_start, work_end in zip(
            range(0, target.shape[0]+1),
            range(self.work_memory_length, target.shape[0]+1)):
            work_anno = target[work_start:work_end]
            self.inputs.append([patient_id, work_start, work_end, work_anno, target])


    def convert_id_to_features(self, cls_ids):
        text_features = torch.cat([self.text_dict[id] for id in cls_ids], dim=0)
        return text_features

    def __getitem__(self, index):
        patient_id, work_start, work_end, work_anno, target = self.inputs[index]
        num_frames = target.shape[0]
        # Get img feature
        if self.modal == 'fusion':
            site_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            roi_feature_path = os.path.join(self.data_path.replace('-cotnet','-SFB-cotnet'),patient_id,'features.npy')
            site_feature = np.load(site_feature_path, mmap_mode='r')
            roi_feature = np.load(roi_feature_path, mmap_mode='r')
            img_feature = np.hstack((site_feature,roi_feature))
        elif self.modal == 'multitask':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        elif self.modal == 'rgb':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        elif self.modal == 'roi':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        else:
            raise(AssertionError)

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        # Get img work memory
        work_img_feature = img_feature[work_indices]
        # Get text work memory
        work_text_anno = copy.deepcopy(work_anno)
        work_text_anno[-1] = 55
        work_text_feature = self.convert_id_to_features(work_text_anno)

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start-1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sampling_rate).clip(0)
            # Get img long memory
            long_img_feature = img_feature[long_indices]
            # Get text long memory
            long_text_anno = target[long_indices]
            long_text_feature = self.convert_id_to_features(long_text_anno)
            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_img_feature = None
            long_text_feature = None
            memory_key_padding_mask = None

        # Get all memory
        if long_img_feature is not None and work_text_feature is not None:
            input_img_feature = np.concatenate((long_img_feature, work_img_feature))
            input_text_feature = np.concatenate((long_text_feature, work_text_feature))
        elif long_img_feature is not None and work_text_feature is None:
            input_img_feature = np.concatenate((long_img_feature, work_img_feature))
            input_text_feature = np.array(long_text_feature)
        else:
            input_img_feature = work_img_feature
            input_text_feature = work_text_feature

        # Convert to tensor
        input_img_feature = torch.as_tensor(input_img_feature.astype(np.float32))
        input_text_feature = torch.as_tensor(input_text_feature.astype(np.float32))
        anno = torch.as_tensor(work_anno)
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (input_img_feature, input_text_feature, memory_key_padding_mask, anno, patient_id, work_indices, num_frames)
        else:
            return (input_img_feature, input_text_feature, anno, patient_id, work_indices, num_frames)


    def __len__(self):
        return len(self.inputs)
    

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)


    def get_target_path(self,patient_name):
        gt_path = './dataset/EGD/foreground_gt/txt'
        element = patient_name.split('/')
        if len(element)>2:
            date_id = element[0]
            gtname = date_id+patient_name.split(element[1])[-1].replace('/','_')
        else:
            gtname = patient_name.replace('/','_')
        return os.path.join(gt_path,gtname+'.txt')

    
class EGDFeature_onlinetest(data.Dataset):
    def __init__(self, config, patient_id):
        if config.seed >= 0:
            basic_utils.set_seed(config.seed)

        self.modal = config.modal
        self.data_path = config.data_path
        self.work_memory_length = config.work_memory_length
        self.long_memory_length = config.long_memory_length
        self.long_memory_sampling_rate = config.long_memory_sampling_rate
        self.long_memory_num_samples = self.long_memory_length//self.long_memory_sampling_rate

        self.inputs = []

        target_path = self.get_target_path(patient_id)
        target = np.array(read_txt(target_path,type='int'))

        for work_start, work_end in zip(
            range(0, target.shape[0]+1),
            range(self.work_memory_length, target.shape[0]+1)):
            work_anno = target[work_start:work_end]
            self.inputs.append([patient_id, work_start, work_end, work_anno, target])

    def __getitem__(self, index):
        patient_id, work_start, work_end, work_anno, target = self.inputs[index]
        num_frames = target.shape[0]
        # Get img feature
        if self.modal == 'fusion':
            site_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            roi_feature_path = os.path.join(self.data_path.replace('-cotnet','-SFB-cotnet'),patient_id,'features.npy')
            site_feature = np.load(site_feature_path, mmap_mode='r')
            roi_feature = np.load(roi_feature_path, mmap_mode='r')
            img_feature = np.hstack((site_feature,roi_feature))
        elif self.modal == 'multitask':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        elif self.modal == 'rgb':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        elif self.modal == 'roi':
            img_feature_path = os.path.join(self.data_path,patient_id,'features.npy')
            img_feature = np.load(img_feature_path, mmap_mode='r')
        else:
            raise(AssertionError)

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        # Get img work memory
        work_img_feature = img_feature[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start-1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sampling_rate).clip(0)
            # Get img long memory
            long_img_feature = img_feature[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_img_feature = None
            memory_key_padding_mask = None
        
        # Get all memory
        if long_img_feature is not None:
            input_img_feature = np.concatenate((long_img_feature, work_img_feature))
        else:
            input_img_feature = work_img_feature

        # Convert to tensor
        input_img_feature = torch.as_tensor(input_img_feature.astype(np.float32))  
        anno = torch.as_tensor(work_anno)
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (input_img_feature, memory_key_padding_mask, anno, patient_id, work_indices, long_indices, num_frames)
        else:
            return (input_img_feature, anno, patient_id, work_indices, long_indices, num_frames)


    def __len__(self):
        return len(self.inputs)
    

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)


    def get_target_path(self,patient_name):
        gt_path = './dataset/EGD/foreground_gt/txt'
        element = patient_name.split('/')
        if len(element)>2:
            date_id = element[0]
            gtname = date_id+patient_name.split(element[1])[-1].replace('/','_')
        else:
            gtname = patient_name.replace('/','_')
        return os.path.join(gt_path,gtname+'.txt')
