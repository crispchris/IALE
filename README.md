# IALE: Imitating Active Learner Ensembles

You can find the paper here: https://www.jmlr.org/papers/v23/21-0387.html

```
@article{JMLR:v23:21-0387,
  author  = {Christoffer Löffler and Christopher Mutschler},
  title   = {IALE: Imitating Active Learner Ensembles},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {107},
  pages   = {1-29},
  url     = {http://jmlr.org/papers/v23/21-0387.html}
}
```

Parts of this code are from [Jordan Ash' BADGE](https://github.com/JordanAsh/badge), [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning) or [Ming Liu's ALIL](https://github.com/Grayming/ALIL).

# Install dependencies
```
conda env create -f environment.yml
```

# Test submitted weights
```
conda activate iale
python active_learn.py
```

# Plot results
```
conda activate iale
jupyter notebook
Experiments.ipynb
```

# Train a policy
```
conda activate iale
python train_policy.py
```

# Visualize datasets
```
conda activate iale
jupyter notebook
Visualize_Datasets.ipynb
```

# Train and test ALIL baseline
```
conda activate py37
cd alil_mnist

# test with included weights for 20 episodes 
python ALIL-transfer.py --output ./experiments --experiment_name bigpolicy_fmnist --dataset_name "fMNIST" --query_strategy alil --policy_path ../weights/alil_20_epi/alil_sim_1000_bigpolicy_MNIST_policy.h5 --annotation_budget 1000 --timesteps 3 --k 10

# train for 100 episodes on MNIST
python ALIL-simulation.py --output ./outputs --dataset_name MNIST --k 10 --annotation_budget 1000 --timesteps 3 --episodes 100
```
