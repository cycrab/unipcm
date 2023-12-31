# MTTOD(modified)

The original code is for the paper "Improving End-to-End Task-Oriented Dialogue System with A Simple Auxiliary Task".
Our work makes a few adjustment for few-shot experiments, also the evaluation scripts of Standardized Evaluation is used in our work. To run UniPCM upon MTTOD, you should download the checkpoint of UniPCM and change the model_path from T5 to corresponding checkpoint.

## checkout source code and data from github repository
To download data.zip properly, git lfs(Large File Storage) extension must be installed.

```
# clone repository as usual
git clone https://github.com/bepoetree/MTTOD.git
cd MTTOD
# check file size of data.zip
ls -l data.zip
# unzip
unzip data.zip -d data/

# The file size of data.zip is about 52 MB. If not, git-lfs is not installed or failed to checked out correctly.
# please ensure to install git-lfs (in Ubuntu or Debian, execute "apt install git-lfs" with sudo) in your system.
# After then, Retrying LFS checkout with the following commands:
git lfs install
git lfs pull
git checkout -f HEAD
```

## Environment setting

Our python version is 3.6.9.

The package can be installed by running the following command.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Preprocessing

For the experiments, we use MultiWOZ2.0 and MultiWOZ2.1.
- (MultiWOZ2.0) annotated_user_da_with_span_full.json: A fully annotated version of the original MultiWOZ2.0 data released by developers of Convlab available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).
- (MultiWOZ2.1) data.json: The original MultiWOZ 2.1 data released by researchers in University of Cambrige available [here](https://github.com/budzianowski/multiwoz/tree/master/data).
- (MultiWOZ2.2) data.json: The MultiWOZ2.2 dataset converted to the same format as MultiWOZ2.1 using script [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2).

We use the preprocessing scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484). Please refer to [here](https://github.com/thu-spmi/damd-multiwoz/blob/master/data/multi-woz/README.md) for the details.

```
python preprocess.py -version $VERSION
```

## Training

Our implementation supports a single GPU. Please use smaller batch sizes if out-of-memory error raises.

- MTTOD without auxiliary task (for the ablation)
```
python main.py -version $VERSION -run_type train -model_dir $MODEL_DIR
python main.py -run_type train -model_dir pretrained_few -add_auxiliary_task -few_shot
# output
```

- MTTOD with auxiliary task
```
python main.py -version $VERSION -run_type train -model_dir $MODEL_DIR -add_auxiliary_task
```

The checkpoints will be saved at the end of each epoch (the default training epoch is set to 10).

## Inference

```
python main.py -run_type predict -ckpt $CHECKPOINT -output $MODEL_OUTPUT -batch_size $BATCH_SIZE
python main.py -run_type predict -ckpt pretrained_new/ckpt-epoch8 -output result.json -batch_size 16
```

All checkpoints are saved in ```$MODEL_DIR``` with names such as 'ckpt-epoch10'.

The result file (```$MODEL_OUTPUT```) will be saved in the checkpoint directory.

To reduce inference time, it is recommended to set large ```$BATCH_SIZE```. In our experiemnts, it is set to 16 for inference.

You can download our trained model [here](https://drive.google.com/file/d/1azIdWPgJKa3PTBFE8lZ1B02bfgguKS2u/view?usp=sharing).

## Evaluation

We use the evaluation scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484).

```
python evaluator.py -data $CHECKPOINT/$MODEL_OUTPUT
```

## Standardized Evaluation

For the MultiWOZ benchmark, we recommend to use [standardized evaluation script](https://github.com/Tomiinek/MultiWOZ_Evaluation).

```
# MultiWOZ2.2 is used for the benchmark (MultiWOZ2.2 should be preprocessed prior to this step)
python main.py -run_type predict -ckpt $CHECKPOINT -output $MODEL_OUTPUT -batch_size $BATCH_SIZE -version 2.2
# convert format for the the standardized evaluation
python convert.py -input $CHECKPOINT/$MODEL_OUTPUT -output $CONVERTED_MODEL_OUTPUT
python convert.py -input pretrained_new/ckpt-epoch8/result.json -output result.json
# clone the standardized evaluation repository
git clone https://github.com/Tomiinek/MultiWOZ_Evaluation
cd MultiWOZ_Evaluation
pip install -r requirements.txt

# do standardized evaluation
python evaluate.py -i $CONVERTED_MODEL_OUTPUT -b -s -r
```
python evaluate.py -i ../result.json -b -s -r

## Acknowledgements

This code is based on the released code (https://github.com/thu-spmi/damd-multiwoz/) for "Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context", which distributed under Apache License Version 2.0. 
Copyright 2019- Yichi Zhang.

For the pre-trained language model, we use huggingface's Transformer (https://huggingface.co/transformers/index.html#), which distributed under Apache License Version 2.0. 
Copyright 2018- The Hugging Face team. All rights reserved.

We are grateful for their excellent works.
