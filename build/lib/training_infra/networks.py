import torch
import numpy as np
import torch.nn as nn

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
    def __init__(self, act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimensions=(10,)):
        super(FCNetwork, self).__init__()
        self.depth = depth
        self.input_size = np.prod(input_dimensions)
        self.output_dimensions = output_dimensions
        self.act = act

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
            x = self.act(layer(x))
        x = self.final_layer(x).view(-1, *self.output_dimensions)
        return x

class CNN(nn.Module):
    def __init__(self, act, width=200, depth = 5, input_dimensions=(1,28,28), output_dimensions=(10,)):
        super(CNN, self).__init__()
        self.depth = depth
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.input_size = np.prod(self.input_dimensions)
        self.act = act
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
            x  = self.act(self.layers[i](x))
            x = self.bns[i](x)
        x = self.final_layer(x).mean(dim=(-1,-2)).view(-1, *self.output_dimensions)
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