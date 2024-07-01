config = {
    'network_width': 100,
    'network_depth': 6,
    'hidden_activation': 'elu',
    'architecture': 'FC',
    'activation': ['circular_activation', 'log_softmax', 'exp_activation'][0],
    'training_loss': ['multi_target_crossentropy_loss', 'neg_sqrt_loss', 'dirichlet_multinomial_loss', 'multinomial_loss'][2],
    'test_loss': ['dirichlet_multinomial_loss', 'multinomial_loss'][0],
    'learning_rate': 1e-4,
    'batch_size': 5000,
    'sampler': ['sequential', 'random'][0],
    'data_augmentation': 'Go',
    'optimizer': 'Adam',
    'training_time' : float('inf')
    }