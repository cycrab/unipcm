# data_utils/dialoglue/top/train.txt
# 64
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/banking/train_10.csv \
--val_data_path dialoglue/data_utils/dialoglue/banking/val.csv \
--test_data_path dialoglue/data_utils/dialoglue/banking/test.csv \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/banking_10_pet_t5/ \
--pet_final pseudo_banking.json \
--train_batch_size 64 --dropout 0.1 --num_epochs 40 --learning_rate 2e-5 \
--model_name_or_path t5-base --task intent --do_lowercase --max_seq_length 512 --dump_outputs \
--generation\
