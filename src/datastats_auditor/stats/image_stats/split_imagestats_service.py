from .base_imagestat_service import BaseImageStatsService
from .base_imagestats import BaseImageStats
from ..io.baseio import IterableDataset
from typing import Literal
from types import SimpleNamespace
from ..entities import SplitStatsResult


SplitImageStatsResult = SimpleNamespace()

class SplitImageStatsComputerService(BaseImageStatsService):
    name = "imagestat_service"
    status = "stable"
    
    
    def __init__(self, image_stats_cls: BaseImageStats, 
                 dataloader_cls: IterableDataset,
                 dataloader_params
                 ):
        self.image_stats_cls = image_stats_cls
        self.dataloader_cls = dataloader_cls
        self.dataloader_params = dataloader_params
    
    def compute_split_image_stats(self): 
        for split_nm, split_param in self.dataloader_params.items():
            print(f"Computing image stats for {split_nm} split...")
            dataloader = self.dataloader_cls(**split_param)
            image_stat = self.image_stats_cls(dataloader)
            imagestat_results = image_stat.compute_image_stats() #compute_dataset_stats(dataset)
            setattr(SplitImageStatsResult, split_nm, imagestat_results)
            print(f"Finished computing image stats for {split_nm} split.")
        return SplitStatsResult(image_stats=SplitImageStatsResult)
