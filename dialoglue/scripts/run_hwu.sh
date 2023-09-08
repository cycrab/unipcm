# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=2 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/hwu/train.json \
--val_data_path dialoglue/data_utils/dialoglue/hwu/valid.json \
--test_data_path dialoglue/data_utils/dialoglue/hwu/test.json \
--token_vocab_path t5-base \
--output_dir checkpoints/hwu/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 30 --learning_rate 1e-4 \
--model_name_or_path t5-base --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
--use_observers \

