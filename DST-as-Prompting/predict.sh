export CUDA_VISIBLE_DEVICES=1
DATA_DIR=../multiwoz/data/MultiWOZ_2.2
cd transformers
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path checkpoints/t5base_mwoz2.2/checkpoint-1000000 \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir /tmp/t5base_mwoz2.2 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column="dialogue" \
    --summary_column="state" \
    --save_steps=50000

