# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/hwu/train_10.json \
--val_data_path dialoglue/data_utils/dialoglue/hwu/valid.json \
--test_data_path dialoglue/data_utils/dialoglue/hwu/test.json \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/hwu_10_pretrain/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 40 --learning_rate 5e-5 \
--model_name_or_path checkpoint4/epoch12 --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
--use_observers --generation\

