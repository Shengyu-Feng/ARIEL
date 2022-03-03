# Adversarial Graph Contrastive Learning with Information Regularization

The official PyTorch implementation of ARIEL in [Adversarial Graph Contrastive Learning with Information Regularization](https://arxiv.org/abs/2202.06491), WWW 2022. Our implementation is based on [GRACE](https://github.com/CRIPAC-DIG/GRACE) and [GCA](https://github.com/CRIPAC-DIG/GCA).


## Dependencies

- torch 1.8.1
- torch-geometric 1.7.0
- sklearn 0.24.2
- numpy 1.20.2
- pyyaml 5.4.1
- networkx 2.5.1

Install all dependencies using
```
pip install -r requirements.txt
```

If you encounter some problems during installing `torch-geometric`, please refer to the installation manual on its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

Train and evaluate the model by executing
```
python train.py --dataset Cora
```
The `--dataset` argument should be one of [ Cora, CiteSeer, AmazonC, AmazonP, CoauthorC, CoauthorP ].
