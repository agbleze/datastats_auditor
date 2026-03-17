import pandas as pd
import uuid
from typing import List
from ..stats.split_service.split_stat_service import SplitStatsComputerService
from ..stats.split_service.base_split_service import BaseSplitStatsComputerService
from ..stats.drift.base_drift_service import BaseDriftComputerService
from ..stats.drift.base_drift import BaseDrift
from ..datacard.datacard_generator import create_data_card
from ..datacard.constants import PDF_PATH, METRICS, FIELD_TO_BIN



def concat_split_dfs(split_dfs: dict):
    df_list = []
    for split_nm, df in split_dfs.items():
        df["split_type"] = split_nm
        df_list.append(df)
        
    split_df = pd.concat(df_list)
    return split_df


def compute_stats_and_drift(split_stats_service: BaseSplitStatsComputerService,
                            drift_stats_service: BaseDriftComputerService,
                            drift_cls: BaseDrift,
                            metrics: List=METRICS, 
                            field_to_bin: List=FIELD_TO_BIN,
                            # compute_spatial_drift=True,
                            # x_coordinate_field="relative_x_center",
                            # y_coordinate_field="relative_y_center",
                            # spatial_strategy="equal",
                            # spatial_n_bins=10,
                            make_date_card: bool = True,
                            **kwargs
                            ):
      
    split_stats_res = split_stats_service.compute_stats()
    
    distributions = split_stats_res.split_dfs    
                
    drift = drift_stats_service(distributions=distributions,
                                drift_cls=drift_cls,
                                metrics=metrics,
                                field_to_bin=field_to_bin
                                )
    
    drift_results = drift.compute_drift_metrics()
    
    if make_date_card:
        create_data_card(split_stats_result=split_stats_res,
                        drift_result=drift_results,
                        version_id=kwargs.get("version_id", str(uuid.uuid1())),
                        name=kwargs.get("name", "demo"),
                        intended_objects=kwargs.get("intended_objects"),
                        pdf_path=kwargs.get("pdf_path", PDF_PATH)
                        )
    return {"split_stats_result": split_stats_res,
            "drift_results": drift_results
            }




#%%

"""computer
            - name
                objectstate
                    params: 
                        ---
                imagestat:
                    params:
                        ---
    
    """
    
    
    