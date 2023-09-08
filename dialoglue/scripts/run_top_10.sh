# data_utils/dialoglue/top/train.txt
#--dump_outputs
CUDA_VISIBLE_DEVICES=0 python dialoglue/run.py                  \
--train_data_path dialoglue/data_utils/dialoglue/top/train_10.txt \
--val_data_path dialoglue/data_utils/dialoglue/top/eval.txt \
--test_data_path dialoglue/data_utils/dialoglue/top/test.txt \
--token_vocab_path t5-base \
--output_dir dialoglue/checkpoints/top_10_prompt_pretrain_25p/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 40 --learning_rate 3e-5 \
--model_name_or_path checkpoint4/epoch14 --task top --do_lowercase --max_seq_length 200 \
--use_observers --generation\

