# data_utils/dialoglue/top/train.txt
#--dump_outputs
CUDA_VISIBLE_DEVICES=2 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/top/train.txt \
--val_data_path dialoglue/data_utils/dialoglue/top/eval.txt \
--test_data_path dialoglue/data_utils/dialoglue/top/test.txt \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/top_gen_ch11_1/ \
--train_batch_size 32 --dropout 0.1 --num_epochs 30 --learning_rate 2e-5 \
--model_name_or_path checkpoint11/epoch14 --task top --do_lowercase --max_seq_length 100 \
--use_observers --generation\

