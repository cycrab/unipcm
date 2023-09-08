# export CUDA_VISIBLE_DEVICES=1,2
DATA_DIR=multiwoz/data/MultiWOZ_2.1
output_dir=transformers/checkpoints/t5pretrain_few_mwoz2.1
#cd ~/DST-as-Prompting

python postprocess.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$DATA_DIR/dummy_few_pretrain/" \
    --test_idx "$DATA_DIR/test.idx" \
    --prediction_txt "$output_dir/generated_predictions.txt"
