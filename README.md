# Transferable Human-object Interaction Detector (THID)

## Installation

Our code is built upon [CLIP](https://github.com/openai/CLIP). This repo require to [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies.

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

## Dataset
The experiments are mainly conducted on HICO-DET and SWIG-HOI dataset. We follow [this repo](https://github.com/YueLiao/PPDM) to prepare the HICO-DET dataset. And we follow [this repo](https://github.com/scwangdyd/large_vocabulary_hoi_detection) to prepare the SWIG-HOI dataset.

## Training
To train a model, run
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch_size 36 --output_dir [path to save checkpoint] --epochs 100 --lr_drop 75 --hoi_token_length 10 --dataset_file swig [or hico]
```

## Inference
To evaluate the trained model, run
```
python main.py --eval --batch_size 8 --hoi_token_length 10 --dataset_file swig [or hico] --pretrained [path to the pretrained model]
```