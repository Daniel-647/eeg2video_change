from .modules.model import VideoDiffusionModel
from .modules.t5 import T5Embedder
from .modules.vae import AutoencoderKL
from .modules.clip import CLIPEmbedder
from .utils.utils import load_checkpoint

class WanT2V:
    def __init__(self, config, checkpoint_dir, device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False):
        self.config = config
        self.device = f'cuda:{device_id}'
        self.rank = rank
        
        # Initialize model components
        self.t5 = T5Embedder(config.t5, fsdp=t5_fsdp)
        self.clip = CLIPEmbedder(config.clip)
        self.vae = AutoencoderKL(config.vae)
        self.model = VideoDiffusionModel(config.model)
        
        # Load checkpoint
        self.model = load_checkpoint(self.model, checkpoint_dir, device_id, rank)
        
    def train_step(self, condition, video):
        """Perform one training step"""
        # Forward pass
        video = video.to(self.device)
        condition = condition.to(self.device)
        
        # Compute loss
        loss = self.model(video, condition)
        return loss