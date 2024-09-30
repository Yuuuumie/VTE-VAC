class Config_egd_lsfit(object):
    def __init__(self, args):

        self.num_classes = args.num_classes
        self.modal = args.modal
        if self.modal == 'fusion':
            self.img_len_feature = 4096
        elif self.modal == 'multitask':
            self.img_len_feature = 2048
        elif self.modal == 'roi':
            self.img_len_feature = 2048
        elif self.modal == 'rgb':
            self.img_len_feature = 2048
        elif self.modal == 'Cholec80':
            self.img_len_feature = 768
        elif self.modal == 'THUMOS14':
            self.img_len_feature = 3072
        elif self.modal == 'I3D':
            self.img_len_feature = 1024
        else:
            raise(AssertionError)
        self.text_len_feature = 768
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.local_rank = args.local_rank
        # loss
        self.lambdas = eval(args.lambdas)
        # lr
        self.lr = args.lr
        self.min_lr = args.min_lr
        self.epochs = args.epochs
        self.lr_cycle_mul = 1.0
        self.lr_cycle_limit = 1
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.lr_noise = []
        self.decay_rate = 0.1
        self.warmup_lr = args.warmup_lr
        self.warmup_epochs = 3
        # model
        self.model_file = args.model_file
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.log_path = args.log_path
        self.model = args.model
        self.seed = args.seed
        self.feature_fps = 25
        # lsfit
        self.num_heads = args.num_heads
        self.dim_feedforward = args.dim_feedforward
        self.img_out_features = 1024
        self.text_out_features = 512 # 512

        self.dropout = args.dropout
        self.activation = args.activation

        self.long_mask_ratio = args.long_mask_ratio
        self.long_memory_length = args.long_memory_length
        self.long_memory_sampling_rate = args.long_memory_sampling_rate
        self.long_memory_num_samples = self.long_memory_length//self.long_memory_sampling_rate

        self.work_mask_ratio = args.work_mask_ratio
        self.work_memory_length = args.work_memory_length
        self.work_memory_sampling_rate = args.work_memory_sampling_rate
        self.work_memory_num_samples = self.work_memory_length//self.work_memory_sampling_rate

        self.enc_module = [[16, 1, True], [32, 2, True]]
        self.dec_module = [-1, 2, True]

        
    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')
