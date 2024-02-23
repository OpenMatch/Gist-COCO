export MODEL_PATH=/data4/LLaMA/llama-7b
export generalize_data=alpaca_plus_validation_human.json
# generalize_data is saved in output_data file.
# MODEL_PATH is Llama
export DATA_TYPE=instruction_compress
#Please set DATA_TYPE to 'passage_compress' when you test Gist-COCO in passage compression scenario,
#and to 'instruction_compress' when you test Gist-COCO in instruction compression scenario.

cd ../src
python -m inference.generalize.generalize_test \
      --model_name $MODEL_PATH \
      --batch_size 8 \
      --input_file ../data/output_data/${generalize_data} \
      --model $MODEL_PATH \
      --data_type $DATA_TYPE \
      --augment_type
        
