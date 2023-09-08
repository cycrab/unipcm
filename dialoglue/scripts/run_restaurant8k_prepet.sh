# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/restaurant8k/train_10.json \
--val_data_path dialoglue/data_utils/dialoglue/restaurant8k/val.json \
--test_data_path dialoglue/data_utils/dialoglue/restaurant8k/test.json \
--token_vocab_path t5-base \
--prepet \
--output_dir dialoglue/checkpoints/rest_10_prepet/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 10 --learning_rate 3e-5 \
--model_name_or_path checkpoint4/epoch14 --task slot --do_lowercase --max_seq_length 100 --dump_outputs \
--use_observers --generation\

