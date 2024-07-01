import torch
import random
import pyro
import torch.nn.functional as F
from train import train
from configs import config
import networks
from activations import hidden_activation_dict, output_activation_dict
from alpharedmond.game_simulator import GameSim

import go_benchmark

dataset_facts = {
    'CIFAR10':{
        'x_dimensions':(3,32,32), 
        'y_dimensions':(10,),
        'target_type': 'class_index'},
    'MNIST':{
        'x_dimensions': (1,28,28), 
        'y_dimensions':(10,),
        'target_type': 'class_index'}, 
    '5x5_go':{
        'x_dimensions': (18,5,5), 
        'y_dimensions':(26, 2),
        'target_type': 'dense_binomial_samples'},
    '3x3_go':{
        'x_dimensions': (18,3,3), 
        'y_dimensions':(10, 2),
        'target_type': 'dense_binomial_samples'}
    }
dataset_name = '3x3_go'
architecture_dict = {
    'FC': networks.FCNetwork,
    'CNN': networks.CNN,
    'BlockCNN': networks.BlockCNN
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
network = architecture_dict[config['architecture']](hidden_act = hidden_activation_dict[config['hidden_activation']],
                                                    output_act = output_activation_dict[config['output_activation']],
                                                    width = config['network_width'],
                                                    depth = config['network_depth'],
                                                    input_dimensions=dataset_facts[dataset_name]['x_dimensions'],
                                                    output_dimensions=dataset_facts[dataset_name]['y_dimensions']
                                                    )
board_dim = 3
gamesim = GameSim(board_dim, None, 'cpu', komi=8.5)
data, root, run_stats, solved, gamesim, visited_nodes = go_benchmark.create_go_dataset(4500000, 30000, board_dimension=3, device=device, gamesim=gamesim)
while True:
    network = train(network.to(device), data, f'{board_dim}x{board_dim}_go_tests', using_wandb=False, config=config).to('cpu')
    model = lambda x: network(x)['alpha']
    # go_benchmark.benchmark_network(model,3)
    data, root, run_stats, solved, gamesim, visited_nodes = go_benchmark.create_go_dataset(45000, 30000, run_stats=run_stats, root=root, network=model, device=device, visited_nodes=visited_nodes, gamesim=gamesim)
    if solved !=0:
        break