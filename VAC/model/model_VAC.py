import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import *
from transformers import AutoTokenizer ,BertModel
import numpy as np

def read_txt(path, type=None):
        txt_list = []
        with open(path) as f:
            for line in f.readlines():
                if not line.find('fushi')>-1:
                    line = line.strip('\n')
                    if type == 'int+':
                        txt_list.append(int(line)+1)
                    elif type == 'int':
                        txt_list.append(int(line))
                    else:
                        txt_list.append(line)
        return txt_list


class Temporal_Channel_Attention(nn.Module):
    def __init__(self, len_feature):
        super(Temporal_Channel_Attention, self).__init__()

        self.len_feature = len_feature
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AvgPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_channels=self.len_feature, out_channels=self.len_feature, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        x = x.permute(1,2,0)
        t_attention_map = self.temporal_attention(x)
        c_attention_map = self.channel_attention(x)

        TC_attention_map = t_attention_map * c_attention_map
        out = x * TC_attention_map
    
        out = out.permute(2,0,1)

        return out


class VAC_encoder(nn.Module):

    def __init__(self, cfg, len_in_feature, len_out_feature):
        super(VAC_encoder, self).__init__()
        self.modal = cfg.modal
        self.len_feature = len_in_feature
        self.d_model = len_out_feature
        self.num_heads = cfg.num_heads
        self.dim_feedforward = cfg.dim_feedforward
        self.dropout = cfg.dropout
        self.activation = cfg.activation
        self.num_classes = cfg.num_classes
        
        # Build long feature heads
        self.long_memory_length = cfg.long_memory_length
        self.long_memory_sampling_rate = cfg.long_memory_sampling_rate
        self.long_memory_num_samples = self.long_memory_length // self.long_memory_sampling_rate
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = nn.Sequential(
                nn.Linear(self.len_feature, self.d_model),
                nn.ReLU(inplace=True),
            )

        # Build work feature head
        self.work_memory_length = cfg.work_memory_length
        self.work_memory_num_samples = self.work_memory_length
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = nn.Sequential(
                nn.Linear(self.len_feature, self.d_model),
                nn.ReLU(inplace=True),
            )

        # Build position encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for param in cfg.enc_module:
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerDecoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerEncoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled:
            param = cfg.dec_module
            dec_layer = TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerDecoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))
        else:
            param = cfg.dec_module
            dec_layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerEncoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))

        # text feature dict
        self.text_dict = {}
        if self.modal == 'Cholec80':
            text_features = np.load("./dataset/Cholec80/Text-features-bert/texts-feature.npy")
        else:
            text_features = np.load("./dataset/EGD/Text-features-bert/texts-feature.npy")
        vocab_size,self.d_text = text_features.shape
        for i in range(vocab_size):
            self.text_dict[i] = torch.tensor(text_features[i:i+1,:])

    
    def convert_id_to_features(self, cls_ids):
        cls_ids = cls_ids.reshape((1,-1)).squeeze(0)
        text_features = torch.cat([self.text_dict[id] for id in cls_ids], dim=0)
        return text_features
    

    def forward(self, input_features, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            long_memories = self.pos_encoding(self.feature_head_long(input_features[:,:self.long_memory_num_samples]).transpose(0, 1))
            
            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]
                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(input_features[:,self.long_memory_num_samples:]).transpose(0, 1)
                                              , padding=self.long_memory_num_samples)

            # Build mask
            mask = generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute element_embeds
            if self.long_enabled:
                element_embeds = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                element_embeds = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )
        return element_embeds
    

class VAC_longText(nn.Module):
    def __init__(self, cfg):
        super(VAC_longText, self).__init__()
        self.fusion_encoder = VAC_encoder(cfg, cfg.img_len_feature+cfg.text_len_feature, cfg.img_out_features)
        self.img_encoder = VAC_encoder(cfg, cfg.img_len_feature, cfg.img_out_features)
        # Build Temporal Channel Attention
        self.TCA = Temporal_Channel_Attention(len_feature=cfg.img_len_feature+cfg.text_len_feature)
        # Build Classifier
        self.classifier_site = nn.Linear(cfg.img_out_features, cfg.num_classes)
        self.classifier_roi = nn.Linear(cfg.img_out_features, 1)
        self.modal = cfg.modal
        # Build text projection
        # self.conv1d = nn.Conv1d(cfg.text_len_feature,cfg.img_len_feature,1,1,0)

    def forward(self, img_features, text_features, memory_key_padding_mask=None):
        if self.modal in ['multitask', 'rgb', 'roi', 'fusion', 'I3D']:
            # # add
            # text_features = self.conv1d(text_features.permute(0,2,1)).permute(0,2,1)
            # fusion_features = text_features + img_features
            # concat
            fusion_features = torch.cat((img_features,text_features),dim=-1)

            # TCA_features = fusion_features
            TCA_features = self.TCA(fusion_features)
            # TCA_features = self.SE(fusion_features)

            TCA_embeds = self.fusion_encoder(TCA_features,memory_key_padding_mask)
            img_embeds = self.img_encoder(img_features,memory_key_padding_mask)
            # Compute classification score
            site_score = self.classifier_site(TCA_embeds).transpose(0, 1)
            roi_score = self.classifier_roi(img_embeds).transpose(0, 1)

            return site_score,roi_score
        
        elif self.modal in ['Cholec80','THUMOS14']:
            fusion_features = torch.cat((img_features,text_features),dim=-1)
            TCA_features = self.TCA(fusion_features)
            TCA_embeds = self.fusion_encoder(TCA_features,memory_key_padding_mask)
            # Compute classification score
            site_score = self.classifier_site(TCA_embeds).transpose(0, 1)

            return site_score


class Model_VAC(nn.Module):
    def __init__(self, cfg):
        super(Model_VAC, self).__init__()
        self.modal = cfg.modal
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.VAC = VAC_longText(cfg)
    
    def forward(self, img_x, text_x, memory_key_padding_mask):
        if self.modal in ['multitask', 'rgb', 'roi', 'fusion','I3D']:
            fg, bg = self.VAC(img_x,text_x,memory_key_padding_mask)
            cas_fuse_softmax = self.softmax(fg)
            bg_proposal = self.sigmoid(bg)
            return cas_fuse_softmax, bg_proposal
    
        elif self.modal in ['Cholec80','THUMOS14']:
            fg = self.VAC(img_x,text_x,memory_key_padding_mask)
            return fg
        
        else:
            raise(AssertionError)