from .base_object_stats_computer_service import BaseObjectStatsComputerService
from .base_object_stats_computer import BaseObjectStatsComputer
from ..io.baseio import BaseAnnotationDFImporter
from types import SimpleNamespace
from ..entities import SplitStatsResult


SplitObjectStatsSummaryResult = SimpleNamespace()
SplitObjectStatsDFResult = SimpleNamespace()

    
class SplitObjectStatsComputerService(BaseObjectStatsComputerService):
    name = "objectstat_service"
    status = "stable"
    
    def __init__(self, objectstats_computer: BaseObjectStatsComputer, 
                 annotation_importer: BaseAnnotationDFImporter,
                 **kwargs
                 ):
        self.object_stats_cls = objectstats_computer
        
        self.annotation_params = {k: v for k, v in kwargs.items() 
                                    if isinstance(v, dict) and list(v.values())[0].endswith("json")
                                    }
        if not self.annotation_params:
            raise ValueError(f"No coco annotation files were provided")
        self.annotation_importer_cls = annotation_importer(**self.annotation_params)
        self.kwargs = kwargs
        
    
    def compute_stats(self): 
        self.splitstat_result = compute_object_stats_per_split(annotation_importer_cls=self.annotation_importer_cls,
                                                                object_stats_cls=self.object_stats_cls,
                                                                **self.kwargs
                                                                )
        return self.splitstat_result
    
    
def compute_object_stats_per_split(annotation_importer_cls,
                                   object_stats_cls,
                                   **kwargs
                                   ):
    coco_dfs = annotation_importer_cls.load()
    summary = {}
    dfs = {}
    for split_nm, coco_df in coco_dfs.items():
        print(f"Computing object stats for {split_nm} split...")
        kwargs["coco_ann_df"] = coco_df
        objstats = object_stats_cls(**kwargs,
                                    )
        objstats_summary = objstats.compute_object_stats()
        summary[split_nm] = objstats_summary
        dfs[split_nm] = objstats.df
        print(f"Finished computing object stats for {split_nm} split.")
        
    splitstat_result = SplitStatsResult(split_dfs=dfs, 
                                        object_stats=summary 
                                        )
    return splitstat_result
    