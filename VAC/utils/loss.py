import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
 
    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):
 
        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
 
    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()
    

class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.lambdas = lambdas
        self.bce_loss = nn.BCELoss()
        self.nll_loss = nn.NLLLoss()
        self.bcelogits_loss = nn.BCEWithLogitsLoss()

    def forward(self, cas_fuse, bg_proposal, image_feat_n, text_feat_n, logit_scale, work_anno):
        point_anno = F.one_hot(work_anno,num_classes=55)[:,:,:-1]
        loss_dict = {}
        # Sites loss
        point_anno_view = point_anno.view(1,-1,point_anno.size(-1))

        cas_softmax_view = cas_fuse.view(1,-1,cas_fuse.size(-1))

        weighting_seq_act, act_label = point_anno_view.max(dim=2, keepdim=True)  # [1,T,1]

        act_idx = torch.nonzero(weighting_seq_act[0,:,0])[:,0].tolist()
        point_anno_act = act_label[:,act_idx,:].squeeze(0).squeeze(-1)

        cas_softmax_act = cas_softmax_view[:,act_idx,:].squeeze(0)
        cas_softmax_act = torch.log(cas_softmax_act + 1e-31)

        loss_frame = self.nll_loss(cas_softmax_act, point_anno_act)
        
        # ROI loss
        bg_label = 1-weighting_seq_act

        bg_label_view = bg_label.view(1,-1,bg_label.size(-1)).squeeze(0).to(torch.float)
        bg_proposal_view = bg_proposal.contiguous().view(1,-1,bg_proposal.size(-1)).squeeze(0)
 
        loss_proposal = self.bce_loss(bg_proposal_view, bg_label_view)
        
        # ITC loss
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * image_feat_n @ text_feat_n.t()

        logits_per_text = logits_per_image.t()
        ITC_labels = self.generate_label_matrix(point_anno_act).cuda()
        loss_itc_i = self.bcelogits_loss(logits_per_image, ITC_labels)
        loss_itc_t = self.bcelogits_loss(logits_per_text, ITC_labels)
        loss_itc = (loss_itc_i+loss_itc_t) / 2
        
        # Total loss
        loss_total = self.lambdas[0]*loss_frame + self.lambdas[1]*loss_proposal + self.lambdas[2]*loss_itc

        # Logger loss dict
        loss_dict["loss_frame"] = loss_frame
        loss_dict["loss_proposal"] = loss_proposal
        loss_dict["loss_itc"] = loss_itc
        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict
    
    def generate_label_matrix(self, lst):
        counts = torch.bincount(lst)  # 获取列表中每个元素的个数
        matrix = torch.zeros(int(lst.size(0)))  # 创建全零矩阵
        for i,count in enumerate(counts):
            sub_matrix = torch.ones((count,count))  # 创建当前元素对应的子矩阵，对角线上为 1
            if i == 0:
                matrix = sub_matrix
            else:
                matrix = torch.block_diag(matrix,sub_matrix)
        return matrix

