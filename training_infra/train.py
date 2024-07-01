import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
import wandb

import traceback
from .networks import architecture_dict
from .configs import train_config
from .activations import output_activation_dict, hidden_activation_dict
from .losses import loss_dict
from torch.utils.data import WeightedRandomSampler, SequentialSampler, BatchSampler
import hydra
import copy

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
    split_stat_string_list = [f'{split} - ' + '  '.join([stat_block.format(stat=stat, value=epoch_stats_dict[split][stat]) for stat in printed_stats if stat in epoch_stats_dict[split]]) for split in epoch_stats_dict]
    print(f'Epoch {epoch} \t' + '  '.join(split_stat_string_list))

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

def hex_augmenter(data):
    x,y = data
    batch_size = len(y)
    board_dim = x.shape[-1]
    num_classes = y.shape[-1]
    board_y = y.view(batch_size, board_dim, board_dim, num_classes)
    rotation = random.randrange(2)
    augmented_x, augmented_board_y = torch.rot90(x, 2*rotation, (-1,-2)), torch.rot90(board_y, 2*rotation, (1,2))
    flip = random.randrange(2)
    if flip:
        augmented_x, augmented_board_y = torch.flip(augmented_x, (-1,)), torch.flip(augmented_board_y, (2,))
    augmented_y = augmented_board_y.reshape(batch_size,-1, num_classes)
    return augmented_x.contiguous(), augmented_y.contiguous()


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
    'AdamW': torch.optim.AdamW,
    'SGD': torch.optim.SGD
    }

sampler_dict = {
    'sequential': lambda y: SequentialSampler(y),
    'random': lambda y: WeightedRandomSampler((y.sum(axis=(-1,-2)).cpu())**0, len(y), replacement=False),
    }

# @profile
def batched_multivariate_hypergeometric(a_batch):
    """
    Draws samples from a multivariate hypergeometric distribution for a batch of input tensors.
    
    Parameters:
    a_batch (torch.Tensor): A 2-dimensional tensor of shape (batch_size, k) where each row
                            contains the number of objects in each category for a batch element.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, k) containing the number of objects drawn from each category.
    """
    a_batch_device = a_batch.device
    a_batch = a_batch.to('cpu')
    batch_size, k = a_batch.shape
    result_batch = torch.zeros_like(a_batch, dtype=torch.int)

    for batch_idx in range(batch_size):
        a = a_batch[batch_idx]
        total_items = a.sum()
        n = torch.randint(0, total_items, (1,)).item()

        if n == 0:
            continue

        remaining_items = a.clone().float().to('cpu')
        remaining_draws = n

        for i in range(k):
            if remaining_draws <= 0:
                break

            if remaining_items[i] == 0:
                continue

            p = remaining_items[i] / remaining_items.sum()
            draw = torch.distributions.Binomial(remaining_draws, p).sample().item()
            draw = min(draw, remaining_items[i].item())

            result_batch[batch_idx, i] = draw
            remaining_draws -= draw
            remaining_items[i] -= draw

    return result_batch.to(a_batch_device)

def conditioning_augmenter(data):
    x,y = data
    target_shape = y.shape
    input_condition = batched_multivariate_hypergeometric(y.flatten(-2,-1)).reshape(target_shape)
    y = y - input_condition
    return (x, input_condition), y
    

data_augmenter_dict = {
    'None': lambda x: x,
    'go': go_augmenter,
    'hex': hex_augmenter
}
# @profile
def train(model, data, dataset_name, using_wandb):
    printed_stats = ['Accuracy', 'Loss']
    if using_wandb:
        wandb.init(project=f"evidential-{dataset_name}")
        config=wandb.config
    else:
        config = train_config
    
    x_train, y_train = data['Train']['x'], data['Train']['y']
    if model == None:
        model = architecture_dict[config['architecture']](hidden_act = hidden_activation_dict[config['hidden_activation']],
                                                          output_act = output_activation_dict[config['output_activation']],
                                                          width = config['network_width'],
                                                          depth = config['network_depth'],
                                                          input_dimensions=x_train.shape[1:],
                                                          output_dimensions=y_train.shape[1:]
                                                          )
    model = model.to(x_train.device)
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=config['learning_rate'])
    training_loss = loss_dict[config['training_loss']]
    test_loss = loss_dict[config['test_loss']]
    data_augmentation = (lambda x: conditioning_augmenter(data_augmenter_dict[config['data_augmentation']](x))) if config['input_conditioning'] else data_augmenter_dict[config['data_augmentation']]
    
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
                x_train_batch, y_train_batch = data_augmentation((x_train[batch_indices], y_train[batch_indices]))
                y_train_batches.append(y_train_batch)
                model_preds = model(x_train_batch)
                model_preds_list.append(model_preds)
                L= training_loss(model_preds, y_train_batch)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            if time.time()>timeout:
                break
            
            epoch_stats_dict = {}
            train_model_preds = {key:torch.vstack([x[key] for x in model_preds_list]) for key in model_preds_list[0]}
            epoch_stats_dict['Train'] = get_split_stats(train_model_preds,torch.cat(y_train_batches,dim=0), test_loss)
            if 'Test' in data:
                test_x, test_y = conditioning_augmenter((data['Test']['x'], data['Test']['y'])) if config['input_conditioning'] else (data['Test']['x'], data['Test']['y'])
                model_preds = model(test_x)
                epoch_stats_dict['Test'] = get_split_stats(model_preds, test_y, test_loss)
                test_loss_value = epoch_stats_dict['Test']['Loss']
                if math.isnan(test_loss_value):
                    break
                # if test_loss_value <.3:
                #     break
                # if test_loss_value >.7:
                #     break
                if test_loss_value < best_models['Loss']['value']:
                    best_models['Loss']['value'] = test_loss_value
                    best_models['Loss']['Model'] = copy.deepcopy(model)
                    if using_wandb:
                        wandb.log({f'Best_Loss': test_loss_value}, step=epoch)
            if using_wandb:
                wandb.log(epoch_stats_dict)
            print_stats(epoch, epoch_stats_dict, printed_stats)

        if using_wandb:
            wandb.finish()
            print('finished wandb')
        return best_models['Loss']['Model'] if 'Test' in data else model
    except KeyboardInterrupt:
        if using_wandb:
            wandb.finish()
            print('finished wandb')
    except Exception:
        traceback.print_exc()
        if using_wandb:
            wandb.finish()
            print('finished wandb')

@hydra.main(config_path=".", config_name="config")
def main(device, data, epochs_per_stats_print, printed_stats, using_wandb, dataset_name, config=None):
    if using_wandb:
        wandb.init(project=f"evidential-{dataset_name}")
        
        config=wandb.config
    model = architecture_dict[config['architecture']](hidden_act = hidden_activation_dict[config['hidden_activation']],
                                                      output_act = output_activation_dict[config['output_activation']],
                                                      width = config['network_width'],
                                                      depth = config['network_depth'],
                                                      input_dimensions=dataset_facts[dataset_name]['x_dimensions'],
                                                      output_dimensions=dataset_facts[dataset_name]['y_dimensions']
                                                      ).to(device)
    
    optimizer = optimizer_dict[config['optimizer']](model.parameters(), lr=config['learning_rate'])
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
                x_train_batch, y_train_batch = data_augmentation((x_train[batch_indices], y_train[batch_indices]))
                y_train_batches.append(y_train_batch)
                model_preds = model(x_train_batch)
                model_preds_list.append(model_preds)
                L= training_loss(model_preds, y_train_batch)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
            if time.time()>timeout:
                break
            
            epoch_stats_dict = {}
            train_model_preds = {key:torch.vstack([x[key] for x in model_preds_list]) for key in model_preds_list[0]}
            epoch_stats_dict['Train'] = get_split_stats(train_model_preds,torch.cat(y_train_batches,dim=0), test_loss)
            model_preds = model(data['Test']['x'])
            epoch_stats_dict['Test'] = get_split_stats(model_preds,data['Test']['y'], test_loss)

            if epoch % epochs_per_stats_print == 0:
                print_stats(epoch, epoch_stats_dict, printed_stats)
            if epoch_stats_dict['Test']['Loss'] <=0:
                print(model_preds)
                print(data['Test']['y'])
                break
            if using_wandb:
                wandb.log(epoch_stats_dict)
            if math.isnan(epoch_stats_dict['Test']['Loss']):
                break
            for stat in ['Loss']:
                stat_value = epoch_stats_dict['Test'][stat]
                if stat_value < best_models[stat]['value']:
                    best_models[stat]['value'] = stat_value
                    architecture = config['architecture']
                    torch.save(model, f'networks/architectures/best_{dataset_name}_{architecture}_architecture.p')
                    torch.save(model.state_dict(), f'networks/weights/best_{dataset_name}_{architecture}_weights.p')
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