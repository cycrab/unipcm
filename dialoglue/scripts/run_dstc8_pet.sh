# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/train_10.json \
--val_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/val.json \
--test_data_path dialoglue/data_utils/dialoglue/dstc8_sgd/test.json \
--token_vocab_path t5-base \
--pet_final dialoglue/pseudo_labels/pseudo_dstc8.json \
--output_dir dialoglue/checkpoints/dstc8_10_pet_pretrain/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 30 --learning_rate 2e-5 \
--model_name_or_path checkpoint8/epoch19 --task slot --do_lowercase --max_seq_length 100 \
--use_observers --generation\

