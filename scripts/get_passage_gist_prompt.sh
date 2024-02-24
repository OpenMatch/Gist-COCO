export COMPRESSION_MODEL=/data/flan-t5/checkpoint-44000
export passage_data=nq_top10.jsonl
#passage_data is popqa, nq, triviaqa or hotpotqa.
# COMPRESSION_MODEL is the path of the checkpoint of Gist-COCO
mkdir -p ../data/output_data
cd ../src
python -m inference.generalize.get_generalize_passage_data \
      --batch_size 8 \
      --input_file ../data/passage/${passage_data} \
      --auxiliary_model $COMPRESSION_MODEL \
      --output_path ../data/output_data/${passage_data} \
      --prompt_k 10
        
