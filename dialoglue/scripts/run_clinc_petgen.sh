# data_utils/dialoglue/top/train.txt
CUDA_VISIBLE_DEVICES=1 python dialoglue/run.py \
--train_data_path dialoglue/data_utils/dialoglue/clinc/train_10.csv \
--val_data_path dialoglue/data_utils/dialoglue/clinc/val.csv \
--test_data_path dialoglue/data_utils/dialoglue/clinc/test.csv \
--pet_data_path dialoglue/data_utils/dialoglue/clinc/train_pet.csv \
--token_vocab_path t5-base \
--petgen \
--output_dir dialoglue/checkpoints/clinc_10_prepet/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 3e-5 \
--model_name_or_path checkpoint8/epoch14 --task intent --do_lowercase --max_seq_length 150 --dump_outputs \
--use_observers --generation\

