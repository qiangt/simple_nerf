# simple_nerf
my own experiments on nerf. I provided both the dataset and the training codes

# environment installation
```
conda env create -f environment.yml
```

# training 
```
python learning_nerf.py
```

# testing
```
python render_360.py
```

# Acknowledge 
I built up these codes based on 
1. tiny nerf colab
```
https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb
```
2. the nerf pytorch
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```