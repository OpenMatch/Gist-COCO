export MODEL_PATH=/data4/flan-t5/flan-t5-large
export CUDA_VISIBLE_DEVICES=0
mkdir -p ../output_check_log
cd ../src/train
# MODEL_PATH is the initial model path of Gist-COCO, we use FlanT5
# output_check_log is used to store checkpoints and logs.
nohup python train_multi_KL.py \
     --output_dir ../../output_check_log/flan-t5 \
     --model_name_or_path $MODEL_PATH  \
     --tokenizer_name google/flan-t5-large \
     --teacher_model_name_or_path $MODEL_PATH  \
     --do_train  \
     --save_steps 2000 \
     --eval_steps 2000 \
     --logging_steps 200 \
     --train_path ../../data/train/train_data.jsonl  \
     --eval_path ../../data/train/dev_data.jsonl  \
     --per_device_train_batch_size 32 \
     --per_device_eval_batch_size 32 \
     --learning_rate 1e-4  \
     --evaluation_strategy steps  \
     --p_max_len 300 \
     --q_max_len 180 \
     --sl_max_len 32  \
     --warmup_ratio 0 \
     --num_train_epochs 8 \
     --fixed_decoder True \
     --train_LM_out_label False \
     --num_prompt 10 \
     --logging_dir ../../output_check_log/flan-t5.log  > ../../output_check_log/flan-t5.out 2>&1 &