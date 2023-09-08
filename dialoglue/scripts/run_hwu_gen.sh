# data_utils/dialoglue/top/train.txt
#--val_data_path dialoglue/data_utils/dialoglue/hwu/valid.json \
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/hwu/train.csv \
--val_data_path dialoglue/data_utils/dialoglue/hwu/val.csv \
--test_data_path dialoglue/data_utils/dialoglue/hwu/test.csv \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/hwu_gen_pretrain_ch11_new/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 30 --learning_rate 3e-5 \
--model_name_or_path checkpoint11/epoch14 --task intent --do_lowercase --max_seq_length 100 --dump_outputs \
--use_observers --generation\

