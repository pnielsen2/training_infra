train_config = {
    'network_width': 175,
    'network_depth': 6,
    'hidden_activation': 'tanh',
    'architecture': 'FC',
    'input_conditioning': False,
    'output_activation': ['circular_activation', 'log_softmax', 'exp_activation'][2],
    'training_loss': ['multi_target_crossentropy_loss', 'neg_sqrt_loss', 'dirichlet_multinomial_loss', 'multinomial_loss', 'epistemic_information_loss', 'aleatoric_information_loss'][2],
    'test_loss': ['dirichlet_multinomial_loss', 'multinomial_loss', 'epistemic_information_loss', 'aleatoric_information_loss'][0],
    'learning_rate': 5e-4,
    'batch_size': 10000,
    'sampler': ['sequential', 'random'][0],
    'data_augmentation': 'None',
    'optimizer': 'AdamW',
    'training_time' : 40
    }

sweep_config = {
    'method': 'bayes',
    'metric':{
        'name': 'Test.Loss',
        'goal': 'minimize'
    },
    'parameters':{
        'architecture': {
            'value': 'FC'
        },
        'network_width':{
            'values':[50, 100, 150, 200, 250, 300, 400, 500, 600, 750]
        },
        'network_depth': {
            'values': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
        },
        'output_activation': {
        'values': ['circular_activation', 'exp_activation']
        },
        'training_loss': {
            'values': ['dirichlet_multinomial_loss']
        },
        'test_loss': {
            'values': ['dirichlet_multinomial_loss']
        },
        'hidden_activation': {
            'values': ['selu', 'relu', 'residual_htanh', 'gelu', 'tanh', 'elu', 'leaky_relu', 'softplus']
        },
        'learning_rate': {
            'values': [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        },
        'batch_size': {
            'values':[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        },
        'sampler': {
            'values': ['sequential', 'random']},
        'data_augmentation': {
            'values': ['None', 'Go']},
        'optimizer': {
            'values': ['Adam', 'SGD']},
        'training_time': {
            'value': 120}
        }
    }