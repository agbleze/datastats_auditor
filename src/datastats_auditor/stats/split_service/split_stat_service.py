
from .base_split_service import BaseSplitStatsComputerService
from datastats_auditor.stats.image_stats.base_imagestat_service import BaseImageStatsService
from datastats_auditor.stats.object_stats.base_object_stats_computer_service import BaseObjectStatsComputerService
from datastats_auditor.stats.image_stats.base_imagestat_service import BaseImageStatsService
from datastats_auditor.stats.object_stats.base_object_stats_computer_service import BaseObjectStatsComputerService
from ..entities import SplitStatsResult
from typing import Union, List


class SplitStatsComputerService(BaseSplitStatsComputerService):
    name = "split_stats_service"
    status = "experimental"
    
    def __init__(self, services_cls: List[Union[BaseImageStatsService,
                                                BaseObjectStatsComputerService
                                                ]
                                          ], 
                 **kwargs
                 ):
        if not isinstance(services_cls, list):
            self.services_cls = [services_cls]
        else:
            self.services_cls = services_cls
            
    def compute_stats(self) -> SplitStatsResult:
        self.split_result = SplitStatsResult()
        for cls in self.services_cls:
            if isinstance(cls, BaseImageStatsService):
                res = cls.compute_split_image_stats()
                self.split_result(image_stats=res.image_stats)
            elif isinstance(cls, BaseObjectStatsComputerService):
                res = cls.compute_stats()
                self.split_result(object_stats=res.object_stats,
                             split_dfs=res.split_dfs
                             )
        return self.split_result
                




    
class SplitStats:
    def __init__(self, object_stats_cls: ObjectStats,
                 image_loader_cls: ImageBatchDataset,
                 object_stats_kwargs: Optional[Dict] = None,
                 image_stats_kwargs: Optional[Dict] = None,
                 **kwargs
                 ):
        self.object_stats_cls = object_stats_cls
        self.image_loader_cls = image_loader_cls
        self.object_stats_kwargs = object_stats_kwargs
        self.image_stats_kwargs = image_stats_kwargs
        
        print(f"self.object_stats_kwargs: {self.object_stats_kwargs}")
        #os.exit(0)
        
        self.imagestat_results = ImageStatsResult()
        self.objectstat_results = ObjectStatsResult()

    def compute_stats(self): 
        for split_nm, split_param in self.image_stats_kwargs.items():
            print(f"Computing image stats for {split_nm} split...")
            dataset = self.image_loader_cls(**split_param)
            imagestat_results = compute_dataset_stats(dataset)
            setattr(self.imagestat_results, split_nm, imagestat_results)
            print(f"Finished computing image stats for {split_nm} split.")

        self.split_df_collection = {}
        for split_nm, split_param in self.object_stats_kwargs.items():
            print(f"Computing object stats for {split_nm} split...")
            #ann_df = coco_annotation_to_df(split_param["ann_file"])
            objstats = self.object_stats_cls(#coco_ann=split_param["ann_file"], 
                                             **split_param
                                             )
            objstats_summary = objstats.summary()
            setattr(self.objectstat_results, split_nm, objstats_summary)
            self.split_df_collection[split_nm] = objstats.df    
            print(f"Finished computing object stats for {split_nm} split.")
        
        #self.object_stats_results = self.objectstat_results
        self.image_stats_results = self.imagestat_results
        self.split_stats = {"image_stats": self.image_stats_results,
                            "object_stats": self.objectstat_results,
                            "split_dfs": self.split_df_collection
                            }
        return self.split_stats
    