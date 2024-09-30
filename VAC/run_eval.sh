data_path="./Features"
model_file="./models/EGD_cotnet_VTEVAC_128_128_4/model_seed_0_24_best.pkl"
output_path="./outputs/EGD_cotnet_VTEVAC_128_128_4"
modal=multitask
num_workers=8
work_memory_length=128
long_memory_length=128
long_memory_sampling_rate=4
num_classes=54

CUDA_VISIBLE_DEVICES=1 python -W ignore ./main_eval/main_eval_EGD.py \
--modal ${modal} --model_file ${model_file} --data_path ${data_path} --work_memory_length ${work_memory_length} \
--long_memory_length ${long_memory_length} --long_memory_sampling_rate ${long_memory_sampling_rate} \
--output_path ${output_path} --num_workers ${num_workers} --num_classes ${num_classes}