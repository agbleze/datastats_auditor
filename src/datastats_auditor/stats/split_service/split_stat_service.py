
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
    
    def __init__(self, 
                 services_cls: List[Union[BaseImageStatsService,
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
                
