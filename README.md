# Selective Partial Domain Adaptation

## Prerequisites

- Python3
- PyTorch ==1.7.1 (with suitable CUDA and CuDNN version)
- torchvision == 0.7.2
- Numpy
- argparse

## Training

```bash
Office-31

python train.py --dst office31 --source amazon_31_list --target dslr_10_list --lr 0.1 --loop-way zip  --epochs 200
```

```bash
Office-Home

python train.py --dst officehome --source Art_list --target Clipart_25_list --lr 0.1 --loop-way zip  --epochs 200
```

```bash
VisDA2017

python train.py --dst visda --source train_list --target val_sub_list --lr 0.1 --loop-way zip  --epochs 200
```

## Acknowledgement

Our code refer the code at: https://github.com/thuml/MDD.

We thank the authors for open sourcing their code.

## Citation

If you use this code for your research, please consider citing:

```

```

## Contact

If you have any problem about our code, feel free to contact [12032913@mail.sustech.edu.cn](mailto:12032913@mail.sustech.edu.cn).

