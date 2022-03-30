# Hierarchical Transformer

Implementation of the ICME2021 paper:

Hierarchical Transformer: Unsupervised Representation Learning for Skeleton-based Human Action Recognition ([PDF]())

*Yi-Bin Cheng, Xipeng Chen, Junhong Chen, Pengxu Wei, Dongyu Zhang, Liang Lin*

# Requirements

- Python3 (>3.5)
- TensorFlow (>1.8)
- tqdm

# Data Preparation

## Datasets

### NTU RGB+D

We follow [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) to process the raw data. Note that we only generate joint-format data.

```
python data_gen/ntu_gen_joint_data.py
```

### PKU-MMD

We borrow codes from [TSMF](https://github.com/bruceyo/TSMF) to process the raw data, and then prepare them as NTU dataset.
 

Transfer the PKU-MMD skeleton data to NTU format
```
python tools/utils/skeleton_to_ntu_format.py
```

After that, generate data like NTU
```
python tools/pku_gendata.py --data_path <path to pku_mmd_skeleton>
```

# Training & Testing

Prepare the training and testing data for the model, e.g.
```
sh script/prepare_ntu_data.sh
```

Run model pre-training, e.g.
```
sh script/run_pretraining_L4.sh
```

Run model finetuening, e.g.
```
sh script/run_finetune_L4.sh
```

# Acknowledgements

This repo is based on

- [BERT](https://github.com/google-research/bert)
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- [TSMF](https://github.com/bruceyo/TSMF)

Thanks to the original authors for their work!

# Citation

```
@inproceedings{Cheng2021hierarchical,
  title={Hierarchical Transformer: Unsupervised Representation Learning for Skeleton-based Human Action Recognition},  
  author={Cheng, Yi-Bin and Chen, Xipeng and Chen, Junhong and Wei, Pengxu and Zhang, Dongyu and Lin, Liang},  
  booktitle={ICME},  
  year={2021},  
}
```