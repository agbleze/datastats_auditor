from .base_object_stats_computer_service import BaseObjectStatsComputerService
from .base_object_stats_computer import BaseObjectStatsComputer
from ..io.baseio import BaseAnnotationDFImporter
from typing import Literal
from types import SimpleNamespace
from ..entities import SplitStatsResult


SplitObjectStatsSummaryResult = SimpleNamespace()
SplitObjectStatsDFResult = SimpleNamespace()

    
class SplitObjectStatsComputerService(BaseObjectStatsComputerService):
    name = "objectstat_service"
    status = "stable"
    
    
    def __init__(self, object_stats_cls: BaseObjectStatsComputer, # not initiated
                 annotation_importer_cls: BaseAnnotationDFImporter, # initiated
                 #annotation_params,
                 #object_stats_params,
                 **kwargs
                 ):
        self.object_stats_cls = object_stats_cls
        self.annotation_importer_cls = annotation_importer_cls
        #self.annotation_params = annotation_params
        #self.object_stats_params = object_stats_params
        self.kwargs = kwargs
        
    
    def compute_stats(self): 
            
        #self.split_df_collection = {}
        # for split_nm, split_annfile in self.annotation_params.items():
        #     print(f"Computing object stats for {split_nm} split...")
        #     #ann_df = coco_annotation_to_df(split_param["ann_file"])
        #     coco_ann_df = self.annotation_importer_cls(split_annfile)
        #     self.object_stats_params["coco_ann_df"] = coco_ann_df
        #     objstats = self.object_stats_cls(#coco_ann=split_param["ann_file"], 
        #                                      **self.object_stats_params
        #                                      )
        #     objstats_summary = objstats.summary()
        #     setattr(SplitObjectStatsSummaryResult, split_nm, objstats_summary)
        #     setattr(SplitObjectStatsDFResult, split_nm, objstats.df)
        #     #self.split_df_collection[split_nm] = objstats.df    
        #     print(f"Finished computing object stats for {split_nm} split.")
        

        # self.splitstat_result = SplitStatsResult(split_dfs=SplitObjectStatsDFResult,
        #                                         object_stats=SplitObjectStatsSummaryResult
        #                                         )
        
        self.splitstat_result = compute_object_stats_per_split(annotation_importer_cls=self.annotation_importer_cls,
                                                                object_stats_cls=self.object_stats_cls,
                                                                **self.kwargs
                                                                )
        return self.splitstat_result
    
    
def compute_object_stats_per_split(annotation_importer_cls,
                                   #object_stats_params, 
                                   object_stats_cls,
                                   **kwargs
                                   ):
    coco_dfs = annotation_importer_cls.load()
    for split_nm, coco_df in coco_dfs.items():
        print(f"Computing object stats for {split_nm} split...")
        #ann_df = coco_annotation_to_df(split_param["ann_file"])
        #coco_ann_df = annotation_importer_cls(split_annfile)
        kwargs["coco_ann_df"] = coco_df
        objstats = object_stats_cls(**kwargs,
                                    #coco_ann=split_param["ann_file"], 
                                    #**object_stats_params
                                    )
        objstats_summary = objstats.summary()
        setattr(SplitObjectStatsSummaryResult, split_nm, objstats_summary)
        setattr(SplitObjectStatsDFResult, split_nm, objstats.df)
        #self.split_df_collection[split_nm] = objstats.df    
        print(f"Finished computing object stats for {split_nm} split.")
        
    splitstat_result = SplitStatsResult(split_dfs=SplitObjectStatsDFResult,
                                        object_stats=SplitObjectStatsSummaryResult
                                        )
    return splitstat_result
    