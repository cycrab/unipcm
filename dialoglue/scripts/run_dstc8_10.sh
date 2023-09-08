# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/train_10.json \
--val_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/val.json \
--test_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/test.json \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/dstc8_10_mp/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 40 --learning_rate 5e-5 \
--model_name_or_path t5-base --task slot --do_lowercase --max_seq_length 100 \
--use_observers --generation\

