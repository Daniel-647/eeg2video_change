from .shared_config import *

WAN_CONFIGS = {
    't2v-1.3B': {
        't5': {
            'model_name': 't5-1.1b',
            'max_length': 128,
            'hidden_size': 768
        },
        'clip': {
            'model_name': 'openai/clip-vit-base-patch32',
            'hidden_size': 512
        },
        'vae': {
            'in_channels': 3,
            'out_channels': 3,
            'hidden_size': 256,
            'latent_channels': 4,
            'num_res_blocks': 2,
            'attention_resolutions': [8, 16],
            'channel_mult': [1, 2, 4, 4],
            'num_heads': 8,
            'dropout': 0.0
        },
        'model': {
            'in_channels': 4,
            'out_channels': 4,
            'hidden_size': 768,
            'context_dim': 768,
            'num_res_blocks': 2,
            'attention_resolutions': [8, 16, 32],
            'channel_mult': [1, 2, 4, 4],
            'num_heads': 8,
            'dropout': 0.1,
            'use_spatial_transformer': True,
            'transformer_depth': 12,
            'use_linear_projection': True,
            'use_checkpoint': True
        }
    }
}