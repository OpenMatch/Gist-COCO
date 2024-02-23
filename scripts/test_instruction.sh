export MODEL_PATH=/data4/flan-t5/flan-t5-large
export COMPRESSION_MODEL=/data3/lixinze/compression_model/task_knowledge_prompt/ACL_compression/open_code_check/flan-t5/checkpoint-44000
export instruction_data=alpaca_plus_validation_unseen.json
#instruction_data is seen, unseen or human.
# MODEL_PATH is FlanT5
# COMPRESSION_MODEL is the path of the checkpoint of Gist-COCO
cd ../src
python -m inference.plug.flan-t5-compression-instruction \
      --batch_size 8 \
      --input_file ../data/instruction/${instruction_data} \
      --auxiliary_model $COMPRESSION_MODEL \
      --model $MODEL_PATH \
      --prompt_k 10 \
      --compression True
        
