import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
import wandb
wandb.login(key='34e59dda416003a2ce211c0c800924effb78801a')
import traceback
import networks
from dataset_loaders import dataset_loaders_dict
from configs import config
from activations import output_activation_dict, hidden_activation_dict
from losses import loss_dict
from torch.utils.data import WeightedRandomSampler, SequentialSampler, BatchSampler


device = torch.device("cuda:0" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
print('device:', device)

# @profile
def get_split_stats(model_preds, y, objective):
    split_dict = {}
    loss = objective(model_preds, y)
    loss = loss.to('cpu').item()
    split_dict.update({'Loss': loss})
    if 'alpha' in model_preds:
        evidence = model_preds['alpha'].sum(dim=-1).mean()
        split_dict.update({'Evidence': evidence.to('cpu').item()})
    if 'n' in model_preds:
        evidence = model_preds['n'].mean()
        split_dict.update({'Evidence': evidence.to('cpu').item()})
    if 'p' in model_preds or 'log_p' in model_preds:
        p_monotone = model_preds['p'] if 'p' in model_preds else model_preds['log_p']
        pred_label = p_monotone.max(1, keepdim=True)[1]
        if len(y.shape) == 1:
            accuracy = (pred_label.eq(y.view_as(pred_label)).sum()/y.shape[0])
            accuracy = accuracy.to('cpu').item(), 
            split_dict.update({'Accuracy': accuracy})
    return split_dict

def print_stats(epoch, epoch_stats_dict, printed_stats):
    stat_block = "{stat}: {value:.4f}"
    split_stat_string_list = [f'{split} - ' + '  '.join([stat_block.format(stat=stat, value=epoch_stats_dict[split][stat]) for stat in printed_stats if stat in epoch_stats_dict[split]]) for split in ['Train', 'Test']]
    print(f'Epoch {epoch} \t' + '  '.join(split_stat_string_list))


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

optimizer_dict = {
    'Adam':torch.optim.Adam,
    'SGD': torch.optim.SGD
    }

architecture_dict = {
    'FC': networks.FCNetwork,
    'CNN': networks.CNN,
    'BlockCNN': networks.BlockCNN
    }

sampler_dict = {
    'sequential': lambda y: SequentialSampler(y),
    'random': lambda y: WeightedRandomSampler((y.sum(axis=(-1,-2)).cpu())**0, len(y), replacement=False),
    }

def go_augmenter(data):
    x,y = data
    batch_size = len(y)
    board_dim = x.shape[-1]
    num_classes = y.shape[-1]
    board_y, pass_y = y[:,:-1,:].view(batch_size, board_dim, board_dim, num_classes), y[:,-1:,:]
    rotation = random.randrange(4)
    augmented_x, augmented_board_y = torch.rot90(x, rotation, (-1,-2)), torch.rot90(board_y, rotation, (1,2))
    flip = random.randrange(2)
    if flip:
        augmented_x, augmented_board_y = torch.flip(augmented_x, (-1,)), torch.flip(augmented_board_y, (2,))
    augmented_y = torch.cat((augmented_board_y.reshape(batch_size,-1, num_classes),pass_y), dim=1)
    return augmented_x.contiguous(), augmented_y.contiguous()

data_augmenter_dict = {
    'None': lambda x: x,
    'Go': go_augmenter
}

# @profile
def train(data, config=None):
    if using_wandb:
        wandb.init(project=f"evidential-{dataset_name}", config=config)
        
        config=wandb.config
    act = config['hidden_activation']
    y_dimensions = dataset_facts[dataset_name]['y_dimensions']
    output_dims = (y_dimensions[0],) if act=='log_softmax' else y_dimensions
    model = architecture_dict[config['architecture']](act = hidden_activation_dict[act],
                                                      width = config['network_width'],
                                                      depth = config['network_depth'],
                                                      input_dimensions=dataset_facts[dataset_name]['x_dimensions'],
                                                      output_dimensions=output_dims
                                                      ).to(device)
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=config['learning_rate'])
    activation = output_activation_dict[config['activation']]
    training_loss = loss_dict[config['training_loss']]
    test_loss = loss_dict[config['test_loss']]
    data_augmentation = data_augmenter_dict[config['data_augmentation']]
    x_train, y_train = data['Train']['x'], data['Train']['y']
    sampler = sampler_dict[config['sampler']](y_train)
    batch_sampler = BatchSampler(sampler, batch_size=config['batch_size'], drop_last=False)

    best_models = {'Loss':{'value': float('inf')},
                   'Accuracy':{'value': 0}}
    
    epoch = 0
    print('Training...')
    try:
        timeout = time.time() + config['training_time']
        while True:
            epoch +=1
            model_preds_list = []
            y_train_batches = []
            for batch_indices in batch_sampler:
                # print(batch_indices)
                x_train_batch, y_train_batch = data_augmentation((x_train[batch_indices], y_train[batch_indices]))
                y_train_batches.append(y_train_batch)
                x = model(x_train_batch)
                model_preds = activation(x)
                # print(model_preds)
                model_preds_list.append(model_preds)
                L= training_loss(model_preds, y_train_batch)
                # print(L)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            if time.time()>timeout:
                break
            
            epoch_stats_dict = {}
            train_model_preds = {key:torch.vstack([x[key] for x in model_preds_list]) for key in model_preds_list[0]}
            epoch_stats_dict['Train'] = get_split_stats(train_model_preds,torch.cat(y_train_batches,dim=0), test_loss)
            x = model(data['Test']['x'])
            model_preds = activation(x)
            epoch_stats_dict['Test'] = get_split_stats(model_preds,data['Test']['y'], test_loss)
            if using_wandb:
                wandb.log(epoch_stats_dict)
            if epoch % epochs_per_stats_print == 0:
                print_stats(epoch, epoch_stats_dict, printed_stats)
            if math.isnan(epoch_stats_dict['Test']['Loss']):
                break
            for stat in ['Loss']:
                stat_value = epoch_stats_dict['Test'][stat]
                if stat_value < best_models[stat]['value']:
                    best_models[stat]['value'] = stat_value
                    architecture = config['architecture']
                    torch.save(model.state_dict(), f'best_{dataset_name}_{architecture}_weights.p')
                    if using_wandb:
                        wandb.log({f'Best_{stat}': stat_value}, step=epoch)
        if using_wandb:
            wandb.finish()
            print('finished wandb')
    except KeyboardInterrupt:
        if using_wandb:
            wandb.finish()
            print('finished wandb')
    except Exception:
        traceback.print_exc()
        if using_wandb:
            wandb.finish()
            print('finished wandb')

loaded_dataset_name = None
dataset_name = '3x3_go'

if dataset_name != loaded_dataset_name:
    print(f'Loading {dataset_name} data into memory...')
    data = dataset_loaders_dict[dataset_name](device)
    print('Loading done')
    loaded_dataset_name = dataset_name

using_wandb = True
printed_stats = ['Accuracy', 'Loss']
epochs_per_stats_print = 1

train(data, config=config)