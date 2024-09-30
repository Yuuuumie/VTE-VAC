CUDA_VISIBLE_DEVICES='0' python3 -m torch.distributed.launch --nproc_per_node=1 generate_multitask_features.py --folder ./cot_experiments/SE-CoTNetD-152_350epoch
