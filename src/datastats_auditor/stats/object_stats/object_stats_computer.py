from .base_object_stats_computer import BaseObjectStatsComputer
from .utils import compute_foreground_area_union
from typing import Literal, Union
import pandas as pd
import numpy as np



class ObjectStatsComputer(BaseObjectStatsComputer):
    name = "objectstats_computer"
    status = "stable"
        
    def __init__(*, self, coco_ann_df: pd.DataFrame, 
                 bins=None, n_bins=5, strategy="quantile", 
                 **kwargs
                 ):
        
        self.df = coco_ann_df.copy()
        self._prepare()
        if not bins:
            bins = self.compute_bins(n_bins=n_bins, strategy=strategy)
        self.bins = bins
        self.area_bin_labels = [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(len(bins)-1)]
        self.df = self.assign_bins(bins, self.area_bin_labels)           
        
    def _prepare(self):
        self.df.dropna(inplace=True)
        self.df["bbox_x"] = self.df["bbox"].apply(lambda b: b[0])
        self.df["bbox_y"] = self.df["bbox"].apply(lambda b: b[1])
        self.df["bbox_w"] = self.df["bbox"].apply(lambda b: b[2])
        self.df["bbox_h"] = self.df["bbox"].apply(lambda b: b[3])
        self.df["image_area"] = self.df["image_width"] * self.df["image_height"]
        self.df["bbox_area"] = self.df["bbox_w"] * self.df["bbox_h"]
        self.df["relative_bbox_area"] = self.df["bbox_area"] / self.df["image_area"] # area of each bbox wrt image area
        self.df["bbox_aspect_ratio"] = self.df["bbox_w"] / self.df["bbox_h"]

        # compute object center coordinates
        self.df["center_x"] = self.df["bbox_x"] + self.df["bbox_w"] / 2
        self.df["center_y"] = self.df["bbox_y"] + self.df["bbox_h"] / 2

        # normalize
        self.df["relative_x_center"] = self.df["center_x"] / self.df["image_width"]
        self.df["relative_y_center"] = self.df["center_y"] / self.df["image_height"]
        
        #self.df["foreground_ratio"] = self.df["bbox_area"] / self.df["image_area"]               
        #self.df["occupancy_per_image"] = self.df.groupby("image_id")["bbox_area"].transform("sum") / self.df["image_area"]
        self.df = compute_foreground_area_union(self.df)
        self.df["occupancy_per_image"] = self.df["foreground_union_area_per_image"] / self.df["image_area"]
        self.df["background_area_per_image"] = self.df["image_area"] - self.df["foreground_union_area_per_image"]
        self.df["foreground_to_background_area_per_image"] = (self.df["foreground_union_area_per_image"]
                                                               / self.df["background_area_per_image"]
                                                               )
        self.df["background_area_norm"] = self.df["background_area_per_image"] / self.df["image_area"]
        self.df["foreground_occupancy_to_background_occupany"] = self.df["occupancy_per_image"] / self.df["background_area_norm"]
        
        num_bboxes = self.df.groupby("image_id").size().rename("num_bboxes_per_image").reset_index()
        self.df = self.df.merge(num_bboxes, on="image_id", how="left")
        relative_bbox_area_var = (self.df.groupby("image_id")
                              ["relative_bbox_area"].var()
                              .fillna(0)
                              .rename("relative_bbox_area_variance_per_image")
                              )
        self.df = self.df.merge(relative_bbox_area_var, on="image_id", how="left")
    
    
    def class_distribution(self):
        counts = self.df["category_name"].value_counts()
        ratios = self.df["category_name"].value_counts(normalize=True).to_dict()
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
        images_per_object = self.df.groupby("category_name").size()
        images_per_object_ratio = images_per_object / images_per_object.sum()
        
        return {
            "object_count": counts.to_dict(),
            "object_ratios": ratios,
            "imbalance_ratio": imbalance_ratio,
            "images_per_object": images_per_object.to_dict(),
            "images_per_object_ratio": images_per_object_ratio.to_dict()
        }
        
    def bbox_geometry(self):
        objects_area_stats = {"mean": self.df.groupby("category_name")["bbox_area"].mean().to_dict(),
                                "median": self.df.groupby("category_name")["bbox_area"].median().to_dict(),
                                "std": self.df.groupby("category_name")["bbox_area"].std().to_dict(),
                                "min": self.df.groupby("category_name")["bbox_area"].min().to_dict(),
                                "max": self.df.groupby("category_name")["bbox_area"].max().to_dict()
                            }
                              
        objects_area_norm_stats = {"mean": self.df.groupby("category_name")["relative_bbox_area"].mean().to_dict(),
                                    "median": self.df.groupby("category_name")["relative_bbox_area"].median().to_dict(),
                                    "std": self.df.groupby("category_name")["relative_bbox_area"].std().to_dict(),
                                    "min": self.df.groupby("category_name")["relative_bbox_area"].min().to_dict(),
                                    "max": self.df.groupby("category_name")["relative_bbox_area"].max().to_dict()
                                }
                                   
        objects_aspect_ratio_stats = {"mean": self.df.groupby("category_name")["bbox_aspect_ratio"].mean().to_dict(),
                                        "median": self.df.groupby("category_name")["bbox_aspect_ratio"].median().to_dict(),
                                        "std": self.df.groupby("category_name")["bbox_aspect_ratio"].std().to_dict(),
                                        "min": self.df.groupby("category_name")["bbox_aspect_ratio"].min().to_dict(),
                                        "max": self.df.groupby("category_name")["bbox_aspect_ratio"].max().to_dict()
                                        }
        
        objects_height = {"mean": self.df.groupby("category_name")["bbox_h"].mean().to_dict(),
                            "median": self.df.groupby("category_name")["bbox_h"].median().to_dict(),
                            "std": self.df.groupby("category_name")["bbox_h"].std().to_dict(),
                            "min": self.df.groupby("category_name")["bbox_h"].min().to_dict(),
                            "max": self.df.groupby("category_name")["bbox_h"].max().to_dict()
                        }
                         
        objects_width = {"mean": self.df.groupby("category_name")["bbox_w"].mean().to_dict(),
                        "median": self.df.groupby("category_name")["bbox_w"].median().to_dict(),
                        "std": self.df.groupby("category_name")["bbox_w"].std().to_dict(),
                        "min": self.df.groupby("category_name")["bbox_w"].min().to_dict(),
                        "max": self.df.groupby("category_name")["bbox_w"].max().to_dict()  
                    }
                         
        
        objects_center_x = {"mean": self.df.groupby("category_name")["center_x"].mean().to_dict(),
                            "median": self.df.groupby("category_name")["center_x"].median().to_dict(),
                            "std": self.df.groupby("category_name")["center_x"].std().to_dict(),
                            "min": self.df.groupby("category_name")["center_x"].min().to_dict(),
                            "max": self.df.groupby("category_name")["center_x"].max().to_dict()
                            }
                            
        
        objects_center_y = {"mean": self.df.groupby("category_name")["center_y"].mean().to_dict(),
                            "median": self.df.groupby("category_name")["center_y"].median().to_dict(),
                            "std": self.df.groupby("category_name")["center_y"].std().to_dict(),
                            "min": self.df.groupby("category_name")["center_y"].min().to_dict(),
                            "max": self.df.groupby("category_name")["center_y"].max().to_dict()
                            }
                            
        objects_relative_x_center = {"mean": self.df.groupby("category_name")["relative_x_center"].mean().to_dict(),
                                "median": self.df.groupby("category_name")["relative_x_center"].median().to_dict(),
                                "std": self.df.groupby("category_name")["relative_x_center"].std().to_dict(),
                                "min": self.df.groupby("category_name")["relative_x_center"].min().to_dict(),
                                "max": self.df.groupby("category_name")["relative_x_center"].max().to_dict()
                                }
                                 
        objects_relative_y_center = {"mean": self.df.groupby("category_name")["relative_y_center"].mean().to_dict(),
                                "median": self.df.groupby("category_name")["relative_y_center"].median().to_dict(),
                                "std": self.df.groupby("category_name")["relative_y_center"].std().to_dict(),
                                "min": self.df.groupby("category_name")["relative_y_center"].min().to_dict(),
                                "max": self.df.groupby("category_name")["relative_y_center"].max().to_dict()
                                }
                        
        
        bbox_stats_area = {
                            "mean": self.df["bbox_area"].mean(),
                            "median": self.df["bbox_area"].median(),
                            "std": self.df["bbox_area"].std(),
                            "min": self.df["bbox_area"].min(),
                            "max": self.df["bbox_area"].max()
                        }
        relative_bbox_area_stats = {"mean": self.df["relative_bbox_area"].mean(),
                                "median": self.df["relative_bbox_area"].median(),
                                "std": self.df["relative_bbox_area"].std(),
                                "min": self.df["relative_bbox_area"].min(),
                                "max": self.df["relative_bbox_area"].max()
                                }
        bbox_stats_aspect_ratio = {"mean": self.df["bbox_aspect_ratio"].mean(),
                                    "median": self.df["bbox_aspect_ratio"].median(),
                                    "std": self.df["bbox_aspect_ratio"].std(),
                                    "min": self.df["bbox_aspect_ratio"].min(),
                                    "max": self.df["bbox_aspect_ratio"].max()
                                }
        bbox_stats_height = {"mean": self.df["bbox_h"].mean(),
                            "median": self.df["bbox_h"].median(),
                            "std": self.df["bbox_h"].std(),
                            "min": self.df["bbox_h"].min(),
                            "max": self.df["bbox_h"].max()
                            }
        bbox_stats_width =  {"mean": self.df["bbox_w"].mean(),
                            "median": self.df["bbox_w"].median(),
                            "std": self.df["bbox_w"].std(),
                            "min": self.df["bbox_w"].min(),
                            "max": self.df["bbox_w"].max()  
                            }      
        bbox_stats_center_x = {"mean": self.df["center_x"].mean(),
                                "median": self.df["center_x"].median(),
                                "std": self.df["center_x"].std(),
                                "min": self.df["center_x"].min(),
                                "max": self.df["center_x"].max()
                            }
        bbox_stats_center_y =   {"mean": self.df["center_y"].mean(),
                                "median": self.df["center_y"].median(),
                                "std": self.df["center_y"].std(),
                                "min": self.df["center_y"].min(),
                                "max": self.df["center_y"].max()
                                }
        bbox_stats_relative_x_center  =   {"mean": self.df["relative_x_center"].mean(),
                                        "median": self.df["relative_x_center"].median(),
                                        "std": self.df["relative_x_center"].std(),
                                        "min": self.df["relative_x_center"].min(),
                                        "max": self.df["relative_x_center"].max()
                                    }
        bbox_stats_relative_y_center =  {"mean": self.df["relative_y_center"].mean(),
                                    "median": self.df["relative_y_center"].median(),
                                    "std": self.df["relative_y_center"].std(),
                                    "min": self.df["relative_y_center"].min(),
                                    "max": self.df["relative_y_center"].max()
                                    }
        
        objects_stats = {"area": objects_area_stats,
                        "area_norm": objects_area_norm_stats,
                        "aspect_ratio": objects_aspect_ratio_stats,
                        "height": objects_height,
                        "width": objects_width,
                        "center_x": objects_center_x,
                        "center_y": objects_center_y,
                        "relative_x_center": objects_relative_x_center,
                        "relative_y_center": objects_relative_y_center
                        }
        bbox_stats = {"aspect_ratio": bbox_stats_aspect_ratio,
                       "area": bbox_stats_area,
                       "relative_area": relative_bbox_area_stats,
                        "height": bbox_stats_height,
                        "width": bbox_stats_width,
                        "center_x": bbox_stats_center_x,
                        "center_y": bbox_stats_center_y,
                        "relative_x_center": bbox_stats_relative_x_center,
                        "relative_y_center": bbox_stats_relative_y_center
                        }
        
        result = {"objects_stats": objects_stats,
                    "bbox_stats": bbox_stats
                }
        return result
    
    def spatial_distribution(self, bins=20):
        heatmap, xedges, yedges = np.histogram2d(self.df["relative_x_center"], 
                                                 self.df["relative_y_center"], 
                                                 bins=bins, range=[[0, 1], [0, 1]]
                                                 )
        heatmap_proba = heatmap / heatmap.sum()
        px = heatmap_proba.sum(axis=1)
        py = heatmap_proba.sum(axis=0)
        res = {
            "heatmap": heatmap,
            "xedges": xedges,
            "yedges": yedges,
            "heatmap_proba": heatmap_proba,
            "px": px,
            "py": py
        }
        return res
    
    def co_occurence(self):
        img_to_classes = (self.df.groupby("image_id")["category_name"]
                          .apply(lambda x: list(set(x)))
                          )
        matrix = pd.crosstab(img_to_classes.index.repeat(img_to_classes.str.len()),
                             np.concatenate(img_to_classes.values)
                             )
        co_matrix = matrix.T.dot(matrix)
        return co_matrix
    
    def difficulty(self, small_object_threshold=0.01,
                   large_object_threshold=0.5
                   ):
        objects_per_image = self.df.groupby("image_id").size()
        avg_objects = objects_per_image.mean()
        min_object_per_image = objects_per_image.min()
        max_object_per_image = objects_per_image.max()
        median_objects_per_image = objects_per_image.median()

        num_imgs = self.df["image_id"].nunique()
        images_per_object = self.df.groupby("category_name")["image_id"].nunique()
        images_per_object_ratio = images_per_object / num_imgs

        small_objects = self.df[self.df["relative_bbox_area"] <= small_object_threshold]
        small_ratio = len(small_objects) / len(self.df)
        large_objects = self.df[self.df["relative_bbox_area"] >= large_object_threshold]
        large_ratio = len(large_objects) / len(self.df)
        medium_objects = self.df[(self.df["relative_bbox_area"] > small_object_threshold) & (self.df["relative_bbox_area"] < large_object_threshold)]
        medium_ratio = len(medium_objects) / len(self.df)

        clutter_score = objects_per_image.mean() / (self.df["image_width"] * self.df["image_height"]).mean()

        foreground_to_background_area_per_image_mean = self.df["foreground_to_background_area_per_image"].mean()
        foreground_to_background_area_per_image_min = self.df["foreground_to_background_area_per_image"].min()
        foreground_to_background_area_per_image_max = self.df["foreground_to_background_area_per_image"].max()
        foreground_to_background_area_per_image_median = self.df["foreground_to_background_area_per_image"].median()
        foreground_to_background_area_per_image_std = self.df["foreground_to_background_area_per_image"].std()



        object_foreground_to_background_area_per_image_mean = self.df.groupby("category_name")["foreground_to_background_area_per_image"].mean().to_dict()
        object_foreground_to_background_area_per_image_max = self.df.groupby("category_name")["foreground_to_background_area_per_image"].max().to_dict()
        object_foreground_to_background_area_per_image_min = self.df.groupby("category_name")["foreground_to_background_area_per_image"].min().to_dict()
        object_foreground_to_background_area_per_image_median = self.df.groupby("category_name")["foreground_to_background_area_per_image"].median().to_dict()
        object_foreground_to_background_area_per_image_std = self.df.groupby("category_name")["foreground_to_background_area_per_image"].std().to_dict()

        bbox_area_bins_ratio = self.df["area_bin_label"].value_counts(normalize=True).to_dict()
        object_bbox_area_per_bins = self.df.groupby(["area_bin_label", "category_name"]).size().unstack(fill_value=0)
        
        occupancy_per_image_mean = self.df["occupancy_per_image"].mean()
        occupancy_per_image_min = self.df["occupancy_per_image"].min()
        occupancy_per_image_max = self.df["occupancy_per_image"].max()
        occupancy_per_image_median = self.df["occupancy_per_image"].median()
        occupancy_per_image_std = self.df["occupancy_per_image"].std()
        
        scene_stats = {"occupancy_per_image": {"mean": occupancy_per_image_mean,
                                                "min": occupancy_per_image_min,
                                                "max": occupancy_per_image_max,
                                                "median": occupancy_per_image_median,
                                                "std": occupancy_per_image_std
                                                }
                         }
        bbox_stats = {"objects_in_image": {"mean": avg_objects,
                                            "min": min_object_per_image,
                                            "max": max_object_per_image,
                                            "median": median_objects_per_image,
                                            "std": objects_per_image.std()
                                            },
                      "foreground_to_background_area_per_image": {"mean": foreground_to_background_area_per_image_mean,
                                            "min": foreground_to_background_area_per_image_min,
                                            "max": foreground_to_background_area_per_image_max,
                                            "median": foreground_to_background_area_per_image_median,
                                            "std": foreground_to_background_area_per_image_std
                                            },
                      
                      "small_object": {"ratio": small_ratio},
                      "medium_object": {"ratio": medium_ratio},
                      "large_object": {"ratio": large_ratio},
                      "bbox_area_ratio_per_bin": bbox_area_bins_ratio,
                      "clutter_score": clutter_score,                      
                      }
        
        object_stats = {"images_per_object_ratio": images_per_object_ratio.to_dict(),
                        "foreground_to_background_area_per_image": {"mean": object_foreground_to_background_area_per_image_mean,
                                            "min": object_foreground_to_background_area_per_image_min,
                                            "max": object_foreground_to_background_area_per_image_max,
                                            "median": object_foreground_to_background_area_per_image_median,
                                            "std": object_foreground_to_background_area_per_image_std
                                            },
                        "bbox_area_ratio_per_bin": object_bbox_area_per_bins.to_dict()
                        }
        
        self.difficulty_metrics ={"bbox_stats": bbox_stats, 
                                  "object_stats": object_stats,
                                  "scene_stats": scene_stats
                                  }
        
        return self.difficulty_metrics
    
    def summary(self):
        class_dist = self.class_distribution()
        bbox_geom = self.bbox_geometry()
        spatial_dist = self.spatial_distribution()
        co_occurence = self.co_occurence()
        difficulty = self.difficulty()
        
        summary = {"class_distribution": class_dist,
                    "bbox_geometry": bbox_geom,
                    "spatial_distribution": spatial_dist,
                    "co_occurence": co_occurence,
                    "difficulty": difficulty
                    }
        return summary
    
    def compute_bbox_area_ratios(self, bins=None, n_bins=None, 
                                 field_name="relative_bbox_area", 
                                 strategy="quantile"):
        areas = self.df[field_name]

        if bins is not None:
            bins = np.array(bins)
        else:
            if n_bins is None:
                n_bins = 5
            if strategy == "quantile":
                bins = np.quantile(areas, np.linspace(0, 1, n_bins + 1))
            elif strategy == "equal":
                bins = np.linspace(0, 1, n_bins +1)
            elif strategy == "log":
                min_area = areas[areas > 0].min()
                bins = np.logspace(np.log10(min_area), 0, n_bins + 1)
            else:
                raise ValueError(f"strategy must be 'quantile', 'equal', or 'log' and not {strategy}")
        labels = [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(len(bins)-1)]
        cat = pd.cut(areas, bins=bins, labels=labels, include_lowest=True)
        counts = cat.value_counts().sort_index()
        ratios = (counts / counts.sum()).sort_index()
        return ratios.to_dict()
    
    def compute_bins(self, n_bins=5, field_name="relative_bbox_area", 
                          strategy="quantile",
                          include_overflow_bin=True
                          ):
        areas = self.df[field_name]
        max_area = areas.max()
        min_area = areas.min()
        if strategy == "quantile":
            bins = np.quantile(areas, np.linspace(0, 1, n_bins + 1))
        elif strategy == "equal":
            bins = np.linspace(min_area, max_area, n_bins +1)
        elif strategy == "log":
            min_area = areas[areas > 0].min()            
            bins = np.logspace(np.log10(min_area), np.log10(max_area), n_bins + 1)
        else:
            raise ValueError(f"strategy must be 'quantile', 'equal', or 'log' and not {strategy}")
        if include_overflow_bin:
            bins = np.concatenate(([-np.inf], bins, [np.inf]))
        return bins
    
    def assign_bins(self, bins, labels, 
                         field_to_bin="relative_bbox_area",
                         name_bin_field_as="area_bin", 
                         name_bin_field_label_as="area_bin_label"
                         ):
        self.df[name_bin_field_as] = pd.cut(self.df[field_to_bin], bins=bins, include_lowest=True)
        self.df[name_bin_field_label_as] = pd.cut(self.df[field_to_bin], bins=bins, labels=labels, include_lowest=True)
        return self.df
    
