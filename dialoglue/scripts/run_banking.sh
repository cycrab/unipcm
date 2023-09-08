# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/banking/train.csv \
--val_data_path dialoglue/data_utils/dialoglue/banking/val.csv \
--test_data_path dialoglue/data_utils/dialoglue/banking/test.csv \
--token_vocab_path t5-base \
--output_dir checkpoints/banking/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 30 --learning_rate 1e-4 \
--model_name_or_path t5-base --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
--use_observers \

