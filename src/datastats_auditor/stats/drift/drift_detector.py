

from .base_drift import BaseDrift
from .drift_utils import (kl_divergence_between_distributions, 
                          js_divergence_between_distributions,
                          compute_spatial_drift
                          )
from ..constant import VALID_BINNING_STRATEGY
import numpy as np
import pandas as pd


class DriftStatsComputer(BaseDrift):
    name = "drift_stats"
    status = "experimental"
     
    def __init__(self, reference_distribution, comparison_distribution,
                 field_to_bin,
                 name_bin_field_as, 
                 name_bin_field_label_as,
                 bins=None,
                 **kwargs
                 ):
        """
        
        kwargs:
            strategy
            include_overflow_bin
            n_bins
            metric
            x_coordinate_field
            y_coordinate_field
        
        """
        self.kwargs = kwargs
        self.reference_distribution = reference_distribution
        self.comparison_distribution = comparison_distribution
        self.bins = bins
        self.field_to_bin = field_to_bin
        self.name_bin_field_as = name_bin_field_as
        self.name_bin_field_label_as = name_bin_field_label_as
        self.strategy = kwargs.get("strategy", "quantile")
        self.include_overflow_bin = kwargs.get("include_overflow_bin", False)
        self.n_bins = kwargs.get("n_bins", 5)
        self.metric = kwargs.get("metric", "js")
        
        if self.strategy not in VALID_BINNING_STRATEGY:
            raise ValueError(f"strategy {self.strategy} is not valid: Valid strategy should be one {VALID_BINNING_STRATEGY}")
        
    
    
    def compute_bins(self, n_bins=None, field_to_bin=None, 
                    strategy=None,
                    ):
        if strategy is None:
            strategy = self.strategy
            
        if field_to_bin is None:
            field_to_bin = self.field_to_bin
        if n_bins is None:
            n_bins = self.n_bins
        #areas = self.df[field_name]#.clip(1e-9, 1.0)
        values = pd.concat([self.reference_distribution[field_to_bin], self.comparison_distribution[field_to_bin]])
        max_value = values.max()
        min_value = values.min()
        if strategy == "quantile":
            bins = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        elif strategy == "equal":
            bins = np.linspace(min_value, max_value, n_bins +1)
        elif strategy == "log":
            min_value = values[values > 0].min()            
            bins = np.logspace(np.log10(min_value), np.log10(max_value), n_bins + 1)
        else:
            raise ValueError(f"strategy must be 'quantile', 'equal', or 'log' and not {strategy}")
        if self.include_overflow_bin:
            bins = np.concatenate(([-np.inf], bins, [np.inf]))
        return bins
    
    def assign_bins(self, distribution,
                    bins, labels, 
                    field_to_bin=None,
                    name_bin_field_as=None, 
                    name_bin_field_label_as=None
                    ):
        if field_to_bin is None:
            field_to_bin = self.field_to_bin
        if name_bin_field_as is None:
            name_bin_field_as = self.name_bin_field_as
        if name_bin_field_label_as is None:
            name_bin_field_label_as = self.name_bin_field_label_as
            
        distribution[name_bin_field_as] = pd.cut(distribution[field_to_bin], bins=bins, include_lowest=True)
        distribution[name_bin_field_label_as] = pd.cut(distribution[field_to_bin], bins=bins, labels=labels, include_lowest=True)
        return distribution
    
    def compute_drift(self, metric="js"):
        if self.bins is not None:
            bins = self.bins
        else:
            bins = self.compute_bins()
        labels = [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(len(bins)-1)]
        self.reference_distribution = self.assign_bins(distribution=self.reference_distribution,
                                                        bins=bins, labels=labels
                                                        )
        self.comparison_distribution = self.assign_bins(distribution=self.comparison_distribution,
                                                        bins=bins, labels=labels
                                                        )
        
        if metric is None:
            metric = self.metric
            
        if metric == "js":
            divergence_res = js_divergence_between_distributions(df1=self.reference_distribution,
                                                                df2=self.comparison_distribution,
                                                                labels=labels,
                                                                field_name=self.name_bin_field_label_as
                                                                )
        if metric == "kl":
            divergence_res = kl_divergence_between_distributions(df1=self.reference_distribution,
                                                                 df2=self.comparison_distribution,
                                                                 labels=labels,
                                                                 field_name=self.name_bin_field_label_as
                                                                 )
            
        return divergence_res
    
    
    def compute_spatial_distribution(self, df,
                                    x_col="relative_x_center", 
                                    y_col="relative_y_center",
                                    **kwargs
                                    ):
        x_col = self.kwargs.get("x_coordinate_field", x_col)
        y_col = self.kwargs.get("y_coordinate_field", y_col)
        x_bins = self.compute_bins(field_to_bin=self.kwargs.get("x_coordinate_field", x_col))
        y_bins = self.compute_bins(field_to_bin=self.kwargs.get("y_coordinate_field", y_col))
    
        heatmap, xedges, yedges = np.histogram2d(df[x_col],
                                                df[y_col],
                                                bins=[x_bins, y_bins],
                                                range=kwargs.get("range", [[0, 1], [0, 1]])
                                            )

        heatmap = heatmap.astype(float)
        total = heatmap.sum()
        if total > 0:
            heatmap /= total

        px = heatmap.sum(axis=1)
        py = heatmap.sum(axis=0)

        return {"heatmap": heatmap,
                "px": px,
                "py": py,
                "xedges": xedges,
                "yedges": yedges,
            }     
        
    @classmethod
    def get_spatial_drift(self, spatial_A, 
                          spatial_B,
                            xy_colname="heatmap",
                            x_colname="px",
                            y_colname="py"
                            ):
        return compute_spatial_drift(spatial_A, spatial_B,
                                    xy_colname=xy_colname,
                                    x_colname=x_colname,
                                    y_colname=y_colname
                                    )










