import pandas as pd
from typing import List
from ..stats.split_service.base_split_service import BaseSplitStatsComputerService
from ..stats.drift.base_drift_service import BaseDriftComputerService
from ..stats.drift.base_drift import BaseDrift
from ..stats.datacard.constants import METRICS, FIELD_TO_BIN
from ..stats.datacard.base_card_creator import BaseCardCreator
from ..stats.datacard.core.io.baseio import BaseCardExporter



def concat_split_dfs(split_dfs: dict):
    df_list = []
    for split_nm, df in split_dfs.items():
        df["split_type"] = split_nm
        df_list.append(df)
        
    split_df = pd.concat(df_list)
    return split_df


def compute_stats_and_drift(split_stats_service: BaseSplitStatsComputerService,
                            drift_stats_service: BaseDriftComputerService,
                            drift_computer: BaseDrift,
                            card_creator: BaseCardCreator,
                            metrics: List=METRICS, 
                            field_to_bin: List=FIELD_TO_BIN,
                            make_date_card: bool = True,
                            card_exporter: BaseCardExporter = None,
                            **kwargs
                            ):
      
    split_stats_res = split_stats_service.compute_stats()
    
    distributions = split_stats_res.split_dfs    
                
    drift = drift_stats_service(distributions=distributions,
                                drift_cls=drift_computer,
                                metrics=metrics,
                                field_to_bin=field_to_bin,
                                **kwargs
                                )
    
    drift_results = drift.compute_drift_metrics()
    
    if make_date_card:
        card_creator = card_creator(split_stats_result=split_stats_res,
                                    drift_result=drift_results,
                                    **kwargs
                                    )
        card_content = card_creator.create_card()
        card_exporter = card_exporter(card_content, 
                                      **kwargs
                                    )
        card_exporter.export()
        
    return {"split_stats_result": split_stats_res,
            "drift_results": drift_results
            }




