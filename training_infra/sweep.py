from configs import sweep_config
import wandb
from train import main
from meta_parameters import meta_parameters
from dataset_loaders import dataset_loaders_dict
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
print('device:', device)


dataset_name = meta_parameters['dataset_name']
print(f'Loading {dataset_name} data into memory...')
data = dataset_loaders_dict[dataset_name](device)
print('Loading done')


sweep_id = wandb.sweep(sweep_config, project=f'evidential-{dataset_name}')
f = lambda config=None: main(device, data, **meta_parameters, config=config)
# wandb.agent(sweep_id, f)
wandb.agent('psikcic6', f)