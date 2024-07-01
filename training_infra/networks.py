import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def get_weight_variances(act, q_star_lower = .1, q_star_upper=5, q_star_guesses = 100, num_simulations = 1000000):
    weight_variances = []
    bias_variances = []
    q_star_list = torch.logspace(np.log10(q_star_lower),np.log10(q_star_upper), q_star_guesses)
    z = torch.randn(num_simulations,requires_grad=True)
    for q_star in q_star_list:
        z.grad=torch.zeros_like(z)
        activations = act(z*q_star**.5)
        activations.sum().backward()
        weight_variance = 1/torch.square(z.grad).mean()
        bias_variance = (q_star - torch.square(activations).mean()*weight_variance)
        weight_variances.append(weight_variance)
        bias_variances.append(bias_variance)
    weight_variances, bias_variances = np.array(weight_variances), np.array(bias_variances)
    allowed_indices = (weight_variances>0)*(bias_variances>0)
    weight_variances, bias_variances = weight_variances[allowed_indices], bias_variances[allowed_indices]
    idx = bias_variances.argmin()
    weight_variance, bias_variance = weight_variances[idx], bias_variances[idx]
    return weight_variance, bias_variance

class FCNetwork(nn.Module):
    def __init__(self, hidden_act, output_act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimensions=(10,)):
        super(FCNetwork, self).__init__()
        self.depth = depth
        self.input_size = np.prod(input_dimensions)
        self.output_dimensions = output_dimensions
        self.hidden_act = hidden_act
        self.output_act = output_act

        self.layers = nn.ModuleList([nn.Linear(self.input_size,width)] + 
                                            [nn.Linear(width,width) for i in range(depth-2)]
        )
        self.final_layer = nn.Linear(width,np.prod(self.output_dimensions))

        # weight_var, bias_var = get_weight_variances(self.act)
        # for l in self.layers:
        #     torch.nn.init.orthogonal_(l.weight, gain=np.sqrt(weight_var/l.weight.shape[-1]))
        #     torch.nn.init.normal_(l.bias.data, std=np.sqrt(bias_var))
        
        # torch.nn.init.orthogonal_(self.final_layer.weight, gain=np.sqrt(1/self.final_layer.weight.shape[-1]))
        # self.final_layer.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(-1,self.input_size)
        for layer in self.layers:
            x = self.hidden_act(layer(x))
        x = self.final_layer(x).view(-1, *self.output_dimensions)
        x = self.output_act(x)
        return x

class ICNetwork(nn.Module):
    def __init__(self, hidden_act, output_act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimensions=(10,)):
        super(ICNetwork, self).__init__()
        self.depth = depth
        self.input_size = np.prod(input_dimensions) + np.prod(output_dimensions)
        self.output_dimensions = output_dimensions
        self.hidden_act = hidden_act
        self.output_act = output_act

        self.layers = nn.ModuleList([nn.Linear(self.input_size,width)] + 
                                            [nn.Linear(width,width) for i in range(depth-2)]
        )
        self.final_layer = nn.Linear(width,np.prod(self.output_dimensions))

        # weight_var, bias_var = get_weight_variances(self.act)
        # for l in self.layers:
        #     torch.nn.init.orthogonal_(l.weight, gain=np.sqrt(weight_var/l.weight.shape[-1]))
        #     torch.nn.init.normal_(l.bias.data, std=np.sqrt(bias_var))
        
        # torch.nn.init.orthogonal_(self.final_layer.weight, gain=np.sqrt(1/self.final_layer.weight.shape[-1]))
        # self.final_layer.bias.data.fill_(0)

    def forward(self, a):
        x, input_condition = a
        x = torch.cat((x.flatten(1,-1), torch.log(1+input_condition.flatten(1,-1))),dim=-1)
        for layer in self.layers:
            x = self.hidden_act(layer(x))
        x = self.final_layer(x).view(-1, *self.output_dimensions)
        x = self.output_act(x)
        return x

class CNN(nn.Module):
    def __init__(self, hidden_act, output_act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimensions=(10,)):
        super(CNN, self).__init__()
        self.depth = depth
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.input_size = np.prod(self.input_dimensions)
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.width = width
        self.channel_size = self.input_dimensions[-1]*self.input_dimensions[-2]
        self.final_layer_size = self.width*self.channel_size

        self.layers = nn.ModuleList([nn.Conv2d(self.input_dimensions[0], width, 3, padding=1)] + 
                                    [nn.Conv2d(width, width, 3,padding=1) for _ in range(depth-2)]
                                    )
        
        self.in_bn = nn.BatchNorm2d(self.input_dimensions[0])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for i in range(depth-1)])
        self.final_layer = nn.Conv2d(width, np.prod(self.output_dimensions), 3)
        
    def forward(self, x):
        x = self.in_bn(x)
        for i in range(self.depth-1):
            x  = self.hidden_act(self.layers[i](x))
            x = self.bns[i](x)
        x = self.final_layer(x).mean(dim=(-1,-2)).view(-1, *self.output_dimensions)
        x = self.output_act(x)
        return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class BlockCNN(nn.Module):
    def __init__(self, act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimension=(10,)):
        super(BlockCNN, self).__init__()
        self.output_dimension = output_dimension
        self.depth = depth
        self.act = act

        self.in_conv = nn.Conv2d(input_dimensions[0], width, 3)
        halfwidth = width//2
        self.ResidBlock = nn.Sequential(
            nn.BatchNorm2d(width),
                                        nn.Conv2d(width,halfwidth,1),
                                        Lambda(self.act),
                                        nn.BatchNorm2d(halfwidth),
                                        nn.Conv2d(halfwidth,halfwidth,3,padding=1),
                                        Lambda(self.act),
                                        nn.BatchNorm2d(halfwidth),
                                        nn.Conv2d(halfwidth,halfwidth,3,padding=1),
                                        Lambda(self.act),
                                        nn.BatchNorm2d(halfwidth),
                                        nn.Conv2d(halfwidth,width,1),
                                        Lambda(self.act)
                                        )
        
        self.layers = nn.ModuleList([self.ResidBlock for _ in range(depth-2)])
        self.in_bn = nn.BatchNorm2d(input_dimensions[0])
        self.out_bn = nn.BatchNorm2d(width)
        self.final_layer = nn.Conv2d(width, output_dimension, 3)
        
    def forward(self, x):
        x = self.in_bn(x)
        x = self.act(self.in_conv(x))
        for block in self.layers:
            x = x + block(x)
        x = self.out_bn(x)
        x = self.final_layer(x).mean(dim=(-1,-2))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoardGameNetwork(nn.Module):
    def __init__(self, hidden_act, output_act, width=256, depth=19, input_dimensions=(18, 5, 5), output_dimensions=(26, 2)):
        super(BoardGameNetwork, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.board_size = input_dimensions[1]  # Assuming square board

        # Determine network structure based on board size
        self.use_conv = self.board_size >= 3
        if self.use_conv:
            self.depth = min(depth, (self.board_size - 1) // 2)  # Ensure we don't reduce dimensions to zero
            self.use_padding = self.board_size >= 5
            padding = 1 if self.use_padding else 0

            # Initial convolutional layer
            self.conv_input = nn.Conv2d(input_dimensions[0], width, kernel_size=3, padding=padding)
            self.bn_input = nn.BatchNorm2d(width)

            # Residual blocks
            self.res_blocks = nn.ModuleList([ResidualBlock(width, hidden_act, use_padding=self.use_padding) for _ in range(self.depth)])

            # Calculate the size after convolutions
            self.conv_output_size = self.board_size - 2 * self.depth * (1 - padding)
        else:
            # For very small boards, use a fully connected network
            self.fc1 = nn.Linear(np.prod(input_dimensions), width)
            self.fc2 = nn.Linear(width, width)
            self.conv_output_size = width

        # Policy head
        policy_input_size = width if not self.use_conv else 2 * self.conv_output_size * self.conv_output_size
        self.policy_fc = nn.Linear(policy_input_size, np.prod(output_dimensions))

        # Value head
        value_input_size = width if not self.use_conv else self.conv_output_size * self.conv_output_size
        self.value_fc1 = nn.Linear(value_input_size, 256)
        self.value_fc2 = nn.Linear(256, 2)  # 2 outputs for beta distribution parameters

    def forward(self, x):
        if self.use_conv:
            # Convolutional pathway
            x = self.hidden_act(self.bn_input(self.conv_input(x)))
            for res_block in self.res_blocks:
                x = res_block(x)
            
            # Policy head
            policy = x
            policy = policy.view(policy.size(0), -1)
            
            # Value head
            value = x
            value = value.view(value.size(0), -1)
        else:
            # Fully connected pathway for very small boards
            x = x.view(x.size(0), -1)
            x = self.hidden_act(self.fc1(x))
            x = self.hidden_act(self.fc2(x))
            policy = value = x

        # Policy output
        policy = self.policy_fc(policy)
        policy = policy.view(-1, *self.output_dimensions)
        policy = self.output_act(policy)

        # Value output
        value = self.hidden_act(self.value_fc1(value))
        value = self.value_fc2(value)
        alpha, beta = F.softplus(value[:, 0]) + 1, F.softplus(value[:, 1]) + 1  # Ensure alpha, beta > 0

        return {'p': policy, 'alpha': alpha, 'beta': beta}

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, activation, use_padding=True):
        super(ResidualBlock, self).__init__()
        self.use_padding = use_padding
        padding = 1 if use_padding else 0
        
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.activation = activation

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if not self.use_padding:
            residual = residual[:, :, 2:-2, 2:-2]
        
        out += residual
        out = self.activation(out)
        return out
    

architecture_dict = {
    'FC': FCNetwork,
    'IC': ICNetwork,
    'CNN': CNN,
    'BlockCNN': BlockCNN,
    'BoardGame': BoardGameNetwork
    }