export COMPRESSION_MODEL=/data/flan-t5/checkpoint-44000
export instruction_data=alpaca_plus_validation_human.json
#instruction_data is seen, unseen or human.
# COMPRESSION_MODEL is the path of the checkpoint of Gist-COCO
mkdir -p ../data/output_data
cd ../src
python -m inference.generalize.get_generalize_instruction_data \
      --batch_size 8 \
      --input_file ../data/instruction/${instruction_data} \
      --auxiliary_model $COMPRESSION_MODEL \
      --output_path ../data/output_data/${instruction_data} \
      --prompt_k 10
        
