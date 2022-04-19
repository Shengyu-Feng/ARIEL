# Adversarial Graph Contrastive Learning with Information Regularization

The official PyTorch implementation of ARIEL in [Adversarial Graph Contrastive Learning with Information Regularization](https://arxiv.org/abs/2202.06491), WWW 2022. Our implementation is based on [GRACE](https://github.com/CRIPAC-DIG/GRACE) and [GCA](https://github.com/CRIPAC-DIG/GCA).


## Updates
- Update the parameters on the lateset `torch-geometric` version, the performance may be a little different. Our results are in folder `results`. Also add the script `batch_train.py` for searching hyperparameters (could launch multiple processes in different terminals) in case new versions come out. 
- Simplify the model with less hyperparameters and searching range. We rescale the range of $`\alpha`$ and change the constraints of $`\beta`$, the searching area becomes smaller now. We currently remove the curriculum learning part with $`\gamma=1`$. Information regularization is not always needed, keep $`\lambda=0`$ at first. In this version, information regularization could improve the performance on `Coauthro-CS`.
- Highly recommend for a larger subgraph size (3000) on large graphs. Will add results for this later.


## Dependencies

- torch 1.10.0
- torch-geometric 2.0.4
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
The `--dataset` argument should be one of [ `Cora`, `CiteSeer`, `AmazonC`, `AmazonP`, `CoauthorC`, `CoauthorP` ].

The parameters in `train.py` are now fixed to be from `config.yaml`, change it to `args.{}` for customer inputs.

## Citation

If you find our work helpful, please cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2202.06491,
  doi = {10.48550/ARXIV.2202.06491},
  
  url = {https://arxiv.org/abs/2202.06491},
  
  author = {Feng, Shengyu and Jing, Baoyu and Zhu, Yada and Tong, Hanghang},
  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Adversarial Graph Contrastive Learning with Information Regularization},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
