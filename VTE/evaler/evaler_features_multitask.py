import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from config import cfg
from datasets import Dataset_individial, create_loader


class Evaler(object):
    def __init__(self,data_config, data_id):
        super(Evaler, self).__init__()
        self.data_id = data_id
        self.loader_eval = self.build_dataset(data_config, data_id)

    def build_dataset(self, data_config, data_id):
        dataset_eval = Dataset_individial(training = False, data_id= data_id)


        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=512,
            is_training=False,
            use_prefetcher=cfg.data_loader.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=cfg.data_loader.workers,
            distributed=cfg.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=cfg.data_loader.pin_mem,
        )
        return loader_eval

    def __call__(self, model, amp_autocast):
        model.eval()

        save_root = os.path.join(
            (os.path.dirname((self.data_id.replace('Videos',f'Features')))),'features.npy')

        if not os.path.exists(os.path.dirname(save_root)):
            os.makedirs(os.path.dirname(save_root))

        with torch.no_grad():
            for batch_idx, data in enumerate(self.loader_eval):
                input, target= data
                if not cfg.data_loader.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                with amp_autocast():
                    feat = model.extract_features(input)
                    feat = feat.data.cpu().numpy()
                    if batch_idx == 0:
                        save_feat = feat
                    else:
                        save_feat = np.vstack((save_feat,save_feat))

            np.save(save_root, save_feat)
