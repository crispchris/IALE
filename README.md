# Install dependencies
```
conda env create -f py37_environment.yml
```

# Test submitted weights
```
conda activate py37
jupyter notebook
execute all cells in run_active_learn.ipynb
```

# Train policy
```
conda activate py37
python train_policy.py
``` 


# Train and test ALIL baseline

The original code authors mentioned in the source files of the ALIL approach are not us, but we modified it.

```
conda activate py37
cd alil_mnist

# test with included weights for 20 episodes 
python ALIL-transfer.py --output ./experiments --experiment_name bigpolicy_fmnist --dataset_name "fMNIST" --query_strategy alil --policy_path ../weights/alil_20_epi/alil_sim_1000_bigpolicy_MNIST_policy.h5 --annotation_budget 1000 --timesteps 3 --k 10

# train for 100 episodes on MNIST
python ALIL-simulation.py --output ./outputs --dataset_name MNIST --k 10 --annotation_budget 1000 --timesteps 3 --episodes 100
```