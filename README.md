# Collaborative-Contrastive-Learning-for-Hypothesis-Domain-Adaptation
The code is released for Hypothesis Domain Adaptation for Speaker Verification


## Environment setup
Note: That is the setting based on my device, you can modify the torch and torchaudio version based on your device.

Memory : We use single GPU(NVIDIA GeForce RTX 4090) and 23~24G RAM in our setting. The batch size depends on your device.

Start from the existing environment
```sh
pip install -r requirement.txt
```


## Data Preparation
Download datasets:

1. CNCeleb (http://cnceleb.org/)
2. Common Voice (zh-TW) (https://commonvoice.mozilla.org/zh-TW/datasets)
3. MUSAN dataset (https://www.openslr.org/17/)
4. RIR dataset (https://www.openslr.org/28/)



# Run Codes

## Training
Note:  You can change the hyperparameters in the trainCHDAModel.py, the following hyperparameters are preset as default values.

Pretrained model is the `exps/pretrain.model` trained on VoxCeleb2.

Every test_step epoches, system will be evaluated in testing set and print the EER.

The result will be saved in `exps/cnceleb1/score.txt`.

The model will saved in `exps/cnceleb1/model`.
### CNCeleb
```sh
python3 trainCHDAModel.py \
  --initial_model exps/pretrain.model \
  --save_path exps/cnceleb1
  --dataset cn \
  --max_epoch 15 \
  --train_list cnceleb/train.csv \
  --train_path CN-Celeb_flac/data \
  --eval_list trials.lst \
  --eval_path CN-Celeb_flac/eval \
  --musan_path musan \
  --rir_path RIRS_NOISES/simulated_rirs \
  --n_class 800
```

### Common Voice
```sh
python3 trainCHDAModel.py \
  --initial_model exps/pretrain.model \
  --save_path exps/common1
  --dataset commonvoice \
  --max_epoch 20 \
  --train_list comonvoice/train.tsv \
  --train_path cv-corpus-13.0-2023-03-09/zh-TW/clips \
  --eval_list comonvoice/eval.tsv \
  --eval_path cv-corpus-13.0-2023-03-09/zh-TW/clips \
  --musan_path musan \
  --rir_path RIRS_NOISES/simulated_rirs \
  --n_class 1641
```


## Evaluation

```sh
python3 trainCHDAModel.py \
    --eval \
    --dataset cn \
    --initial_model exps/pretrain.model
```
