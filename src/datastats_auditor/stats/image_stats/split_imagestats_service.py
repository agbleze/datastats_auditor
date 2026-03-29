from .base_imagestat_service import BaseImageStatsService
from .base_imagestats import BaseImageStats
from ..io.baseio import IterableDataset
from types import SimpleNamespace
from ..entities import SplitStatsResult
import os


SplitImageStatsResult = SimpleNamespace()

class SplitImageStatsComputerService(BaseImageStatsService):
    name = "imagestat"
    status = "stable"
    
    def __init__(self, imagestat_computer: BaseImageStats, 
                 dataloader: IterableDataset,
                 **kwargs
                 ):
        self.image_stats_cls = imagestat_computer
        self.dataloader_cls = dataloader
        
        self.dataloader_params = {k: v for k, v in kwargs.items() 
                                    if isinstance(v, dict) and os.path.isdir(list(v.values())[0])
                                    }
        if not self.dataloader_params:
            raise ValueError(f"Image directory was not provided")
    
    def compute_split_image_stats(self): 
        for split_nm, split_param in self.dataloader_params.items():
            print(f"Computing image stats for {split_nm} split...")
            dataloader = self.dataloader_cls(**split_param)
            image_stat = self.image_stats_cls(dataloader)
            imagestat_results = image_stat.compute_image_stats() 
            setattr(SplitImageStatsResult, split_nm, imagestat_results)
            print(f"Finished computing image stats for {split_nm} split.")
        return SplitStatsResult(image_stats=SplitImageStatsResult)