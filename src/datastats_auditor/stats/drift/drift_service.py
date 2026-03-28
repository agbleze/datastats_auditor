from .base_drift_service import BaseDriftComputerService
from .base_drift import BaseDrift
from ...stats.utils import (plot_spatial_heatmaps,
                            get_drift_result_as_df,
                            plot_drift_radar
                            )
from typing import Dict, List, Union
import pandas as pd
from itertools import combinations


class DriftComputerService(BaseDriftComputerService):
    name = "drift_computer_service"
    status = "experimental"
    
    def __init__(self, drift_cls: BaseDrift, 
                 distributions: Dict[str, pd.DataFrame],
                metrics: Union[str, List[str]],
                field_to_bin: Union[str, List[str]],
                **kwargs
                ):
        """
        
        kwargs:
            "strategy"
            "n_bins"
            "bins"
            "include_overflow_bin"
            "compute_spatial_drift": bool
            "x_coordinate_field"
            "y_coordinate_field"
            
            
        """
        self.drift_cls = drift_cls
        self.kwargs = kwargs
        self.distributions = distributions
        self.metrics = metrics
        self.distribution_pairs = list(combinations(distributions.keys(), 2))
        self.strategy = self.kwargs.get("strategy", "quantile")
        self.n_bins = self.kwargs.get("n_bins", 5)
        self.bins = self.kwargs.get("bins")
        self.include_overflow_bin = self.kwargs.get("include_overflow_bin", False)
        
        for i in [field_to_bin, metrics]:
            if not isinstance(i, (str, list)):
                raise TypeError(f"{i} needs to be of type str or list and not {type(i)}")
        
        if isinstance(field_to_bin, str):
            self.field_to_bin = [field_to_bin]
        else:
            self.field_to_bin = field_to_bin
        if isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = metrics
            
    def compute_drift_metrics(self, plot_metric_name="js"):
        self.drift_results = {}
        self.spatial_drift_result = {}
        self.spatial_distribution = {}
        self.spatial_heatmap = {}
        self.drift_plot = {}
        
        for pair in self.distribution_pairs:
            ref, comp = pair
            ref_df = self.distributions[ref]
            comp_df = self.distributions[comp]
            
            if self.kwargs.get("compute_spatial_drift", False):
                x_coordinate_field = self.kwargs.get("x_coordinate_field")
                y_coordinate_field = self.kwargs.get("y_coordinate_field")
                
                spatial_drift = self.drift_cls(reference_distribution=ref_df,
                                                comparison_distribution=comp_df,
                                                field_to_bin=None,
                                                name_bin_field_as=self.kwargs.get("spatial_name_bin_field_as"),
                                                name_bin_field_label_as=self.kwargs.get("spatial_name_bin_field_label_as"),
                                                strategy=self.kwargs.get("spatial_strategy", self.strategy),
                                                n_bins=self.kwargs.get("spatial_n_bins"),
                                                )
                for match in pair:
                    distr = spatial_drift.compute_spatial_distribution(df=self.distributions[match],
                                                                        x_col=x_coordinate_field,
                                                                        y_col=y_coordinate_field
                                                                        )
                    self.spatial_distribution[match] = distr
                    
                self.spatial_heatmap[pair] = plot_spatial_heatmaps(spatial_dict_A=self.spatial_distribution[pair[0]],
                                                                    spatial_dict_B=self.spatial_distribution[pair[1]],
                                                                    names=pair
                                                                    )
                
                self.spatial_drift_result[pair] = self.drift_cls.get_spatial_drift(self.spatial_distribution[pair[0]],
                                                                                self.spatial_distribution[pair[1]],
                                                                                )
                
            for field in self.field_to_bin:
                name_bin_field_as = f"{field}_bin"
                name_bin_field_label_as = f"{name_bin_field_as}_label"
                for metric in self.metrics:
                    drift_cls = self.drift_cls(reference_distribution=ref_df,
                                                comparison_distribution=comp_df,
                                                field_to_bin=field,
                                                name_bin_field_as=name_bin_field_as,
                                                name_bin_field_label_as=name_bin_field_label_as,
                                                bins=self.bins, n_bins=self.n_bins,
                                                strategy=self.strategy,
                                                metric=metric
                                                )     
                    drift_res = drift_cls.compute_drift()  
                    self.drift_results[f"{pair}_{field}_{metric}"] = {metric: drift_res,
                                                                    "distribution_pair": pair,
                                                                    "property": field
                                                                    }  
        
        for pair in self.distribution_pairs:
            df = get_drift_result_as_df(drift_results=self.drift_results,
                                        distribution_pair=str(pair),
                                        property_field_name="property",
                                        metric_name=plot_metric_name,
                                        )
            
            title = f"{plot_metric_name}".upper()
            title = f"{title} Divergence between {pair} Distribution"
            plot = plot_drift_radar(drift_df=df, drift_properties_colname="property",
                                    drift_scores_colname="scores",
                                    title=title, height=self.kwargs.get("height"),
                                    width=self.kwargs.get("width")
                                    )
            self.drift_plot[pair] = plot
            
        return {"drift": self.drift_results, 
                "spatial_drift": self.spatial_drift_result,
                "spatial_distribution": self.spatial_distribution,
                "spatial_heatmap": self.spatial_heatmap,
                "drift_plot": self.drift_plot
                }    