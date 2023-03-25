# Selective Partial Domain Adaptation

This is the official Pytorch implementation of [Selective Partial Domain Adaptation](https://bmvc2022.mpi-inf.mpg.de/420/).

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

## Citation

If you use this code for your research, please consider citing:

```
@inproceedings{guo2022selective,
  title={Selective Partial Domain Adaptation},
  author={Guo, Pengxin and Zhu, Jinjing and Zhang, Yu},
  booktitle={33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
  publisher={{BMVA} Press},
  year={2022},
  url={https://bmvc2022.mpi-inf.mpg.de/0420.pdf}
}
```

## Acknowledgement

Our code refer the code at: https://github.com/thuml/MDD.

We thank the authors for open sourcing their code.


## Contact

If you have any problem about our code, feel free to contact [12032913@mail.sustech.edu.cn](mailto:12032913@mail.sustech.edu.cn).

