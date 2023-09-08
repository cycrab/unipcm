# export CUDA_VISIBLE_DEVICES=1,2
DATA_DIR=multiwoz/data/MultiWOZ_2.1
output_dir=transformers/checkpoints/t5pretrain_few_mwoz2.1
#cd ~/DST-as-Prompting

python eval.py \
    --data_dir "$DATA_DIR" \
    --prediction_dir "$DATA_DIR/dummy_few_pretrain/" \
    --output_metric_file "$DATA_DIR/dummy_few_pretrain/prediction_score"
