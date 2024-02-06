# ARIEL: Adversarial Graph Contrastive Learning

The official PyTorch implementation of ARIEL in [Adversarial Graph Contrastive Learning with Information Regularization](https://arxiv.org/abs/2202.06491), WWW 2022, and its extended version [ARIEL: Adversarial Graph Contrastive Learning](https://arxiv.org/abs/2208.06956v1) accepted to TKDD. Our implementation is based on [GRACE](https://github.com/CRIPAC-DIG/GRACE), [GCA](https://github.com/CRIPAC-DIG/GCA) and [GraphCL](https://github.com/Shen-Lab/GraphCL).


## Updates (Nov., 2023)
- Release the code and data for graph contrastive learning under the poisoning attack.
- Add two additional node classification datasets: `Facebook` and `LastFMAsia`.
- Release the code for the graph classification task.   
- If you want to contact me for any question, please note the change of the email addrees, the new address is: shengyuf@andrew.cmu.edu


## Updates (Aug., 2022)
- Update the parameters on the lateset `torch-geometric` version, the performance may be a little different. Our results are in folder `results`. Also add the script `batch_train.py` for searching hyperparameters (could launch multiple processes in different terminals) in case new versions come out. 
- Simplify the model with less hyperparameters and searching range. We rescale the range of `alpha` and change the constraints of `beta`, the searching area becomes smaller now. We currently remove the curriculum learning part with `gamma=1`. Information regularization is not always needed, keep `lamb=0` at first. In this version, information regularization could improve the performance on `Coauthro-CS`.
- Highly recommend for a larger subgraph size (3000) on large graphs. 


## Setup

```
conda env create -f environment.yml
conda activate ARIEL
```

If you encounter any problem during installing `torch-geometric`, please refer to the installation manual on its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

### Node Classification
Train and evaluate the model by executing
```
cd node_cls
python train.py --dataset Cora
```
The `--dataset` argument should be one of `Cora, CiteSeer, AmazonC, AmazonP, CoauthorC, CoauthorP, Facebook, LastFMAsia`.

The parameters in `train.py` are now fixed to be from `config.yaml`, change it to `args.{}` for customer inputs.

For the node classification under the poisoning attack, change the script from `train.py` to `train_attack.py`, the [attacked graphs](https://drive.google.com/file/d/1EHlM1O92_YuqxcPYchbFovGRc5EZcQJV/view?usp=sharing) should be put in the folder `node_cls/attack_data`.

### Graph Classification
Train and evaluate the model by executing
```
cd graph_cls
bash go.sh $GPU_ID $DATASET_NAME $AUGMENTATION
```

`$DATASET_NAME` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), `$GPU_ID` is the lanched GPU ID and `$AUGMENTATION` could be `random2, random3, random4` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately. By default, we use `random4` as the augmentation in our paper.

## Citation

If you find our work helpful, please cite our paper:

```
@inproceedings{feng2022adversarial,
    author = {Feng, Shengyu and Jing, Baoyu and Zhu, Yada and Tong, Hanghang},
    title = {Adversarial Graph Contrastive Learning with Information Regularization},
    year = {2022},
    isbn = {9781450390965},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3485447.3512183},
    doi = {10.1145/3485447.3512183},
    location = {Virtual Event, Lyon, France},
    series = {WWW '22}
}

@misc{feng2022ariel,
      title={ARIEL: Adversarial Graph Contrastive Learning}, 
      author={Shengyu Feng and Baoyu Jing and Yada Zhu and Hanghang Tong},
      year={2022},
      eprint={2208.06956},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
