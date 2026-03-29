from .base_imagestats import BaseImageStats
from .utils import compute_dataset_stats


class ImageStatsComputer(BaseImageStats):
    name = "imagestats"
    status = "experimental"
    
    def __init__(self, dataloader, **kwargs):
        self.dataloader = dataloader
        
    def compute_image_stats(self, *args, **kwargs):
        result = compute_dataset_stats(self.dataloader)
        return result  