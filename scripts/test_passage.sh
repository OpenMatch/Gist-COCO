export MODEL_PATH=/data4/flan-t5/flan-t5-large
export COMPRESSION_MODEL=/data/flan-t5/checkpoint-44000
export passage_data=nq_top10.jsonl
#passage_data is popqa, nq, triviaqa or hotpotqa.
# MODEL_PATH is FlanT5
# COMPRESSION_MODEL is the path of the checkpoint of Gist-COCO
cd ../src
python -m inference.plug.flan-t5-compression-passage \
      --batch_size 8 \
      --input_file ../data/passage/${passage_data} \
      --auxiliary_model $COMPRESSION_MODEL \
      --model $MODEL_PATH \
      --prompt_k 10 \
      --compression True
        
