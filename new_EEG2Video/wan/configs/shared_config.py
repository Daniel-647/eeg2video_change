# Shared configuration for Wan models

# Common model configurations
MODEL_CONFIG = {
    'hidden_size': 768,
    'num_heads': 8,
    'dropout': 0.1,
    'use_checkpoint': True
}

# Common T5 configurations
T5_CONFIG = {
    'model_name': 't5-1.1b',
    'max_length': 128,
    'hidden_size': 768
}

# Common CLIP configurations
CLIP_CONFIG = {
    'model_name': 'openai/clip-vit-base-patch32',
    'hidden_size': 512
}

# Common VAE configurations
VAE_CONFIG = {
    'in_channels': 3,
    'out_channels': 3,
    'hidden_size': 256,
    'latent_channels': 4,
    'num_res_blocks': 2,
    'attention_resolutions': [8, 16],
    'channel_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'dropout': 0.0
}