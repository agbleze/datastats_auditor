

#%%
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from pandas import json_normalize
import json
from dataclasses import dataclass
from typing import Dict, Union, List, Optional
from datastats_auditor.stats.image_stats import ImageBatchDataset
from datastats_auditor.stats.image_stats import compute_dataset_stats, estimate_image_memory_size_GB, get_memory_info
from itertools import combinations
from shapely.geometry import box
from shapely.ops import unary_union
import plotly.express as px

#%%

VALID_BINNING_STRATEGY = ["quantile", "equal", "log"]

def coco_annotation_to_df(coco_annotation_file):
    with open(coco_annotation_file, "r") as annot_file:
        annotation = json.load(annot_file)
    annotations_df = json_normalize(annotation, "annotations")
    annot_imgs_df = json_normalize(annotation, "images")
    annot_cat_df = json_normalize(annotation, "categories")
    annotations_images_merge_df = annotations_df.merge(annot_imgs_df, left_on='image_id', 
                                                        right_on='id',
                                                        suffixes=("_annotation", "_image"),
                                                        how="outer"
                                                        )
    annotations_imgs_cat_merge = annotations_images_merge_df.merge(annot_cat_df, left_on="category_id", right_on="id",
                                                                    suffixes=(None, '_categories'),
                                                                    how="outer"
                                                                    )
    all_merged_df = annotations_imgs_cat_merge[['id_annotation', 'image_id','category_id', 'bbox', 'area', 'segmentation', 'iscrowd',
                                'file_name', 'height', 'width', 'name', 'supercategory'
                                ]]
    all_merged_df.rename(columns={"name": "category_name",
                                  "height": "image_height",
                                  "width": "image_width"}, 
                         inplace=True
                         )
    all_merged_df.dropna(subset=["file_name"], inplace=True)
    return all_merged_df


def compute_foreground_area_union(df):
    def bbox_area_union(group):
        polys = []
        for _, row in group.iterrows():
            x_min = row["bbox_x"]
            y_min = row["bbox_y"]
            x_max = row["bbox_x"] + row["bbox_w"]
            y_max = row["bbox_y"] + row["bbox_h"]
            polys.append(box(x_min, y_min, x_max, y_max))

        if not polys:
            return 0.0

        union_poly = unary_union(polys)
        return union_poly.area 
    ratios = (
        df.groupby("image_id")
          .apply(bbox_area_union)
          .rename("foreground_union_area_per_image")
          .reset_index()
    )

    return df.merge(ratios, on="image_id")

#%%    

train_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/train/_annotations.coco.json"
val_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/valid/_annotations.coco.json"
test_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/test/_annotations.coco.json"

train_annot_df = coco_annotation_to_df(train_annfile)

#%%
## Class Distribution Analysis
#%% count per class 
train_annot_df.category_name.value_counts()

#%% class imbalance ratio
train_annot_df.category_name.value_counts(normalize=True) * 100

#%% ###### Bounding Box Geometry Analysis ######
train_annot_df.dropna(inplace=True)
train_annot_df["bbox_x"] = train_annot_df["bbox"].apply(lambda b: b[0])
train_annot_df["bbox_y"] = train_annot_df["bbox"].apply(lambda b: b[1])
train_annot_df["bbox_w"] = train_annot_df["bbox"].apply(lambda b: b[2])
train_annot_df["bbox_h"] = train_annot_df["bbox"].apply(lambda b: b[3])

train_annot_df["bbox_area"] = train_annot_df["bbox_w"] * train_annot_df["bbox_h"]
train_annot_df["relative_bbox_area"] = train_annot_df["bbox_area"] / (train_annot_df["image_width"] * train_annot_df["image_height"])
train_annot_df["bbox_aspect_ratio"] = train_annot_df["bbox_w"] / train_annot_df["bbox_h"]


# %% #### Spatial Distribution Analysis ######
# compute object center coordinates
train_annot_df["center_x"] = (train_annot_df["bbox_x"] + train_annot_df["bbox_w"]) / 2
train_annot_df["center_y"] = (train_annot_df["bbox_y"] + train_annot_df["bbox_h"]) / 2

# normalize
train_annot_df["relative_x_center"] = train_annot_df["center_x"] / train_annot_df["image_width"]
train_annot_df["center_y_norm"] = train_annot_df["center_y"] / train_annot_df["image_height"]



#%%  ####### Class Co-Occurence Matrix ######

co_occurence_ex = pd.crosstab(train_annot_df["image_id"], train_annot_df["category_name"])

co_occurence = (train_annot_df.groupby("image_id")["category_name"]
                .apply(lambda x: list(set(x)))
                .explode()
                .reset_index()
                )

matrix = pd.crosstab(co_occurence["image_id"], co_occurence["category_name"])
matrix.T.dot(matrix) #- np.diag(matrix.sum(axis=0))


#%%  ##### Dataset Difficulty Metrics ######
objects_per_image = train_annot_df.groupby("image_id").size()

avg_objects = objects_per_image.mean()
avg_objects

num_imgs = train_annot_df["image_id"].nunique()
images_per_object = train_annot_df.groupby("category_name")["image_id"].nunique()
images_per_object_ratio = images_per_object / num_imgs

small_objects = train_annot_df[train_annot_df["relative_bbox_area"] < 0.01]
small_ratio = len(small_objects) / len(train_annot_df)
small_ratio

clutter_score = objects_per_image.mean() / (train_annot_df["image_width"] * train_annot_df["image_height"]).mean()
clutter_score

train_annot_df["foreground_ratio"] = train_annot_df["bbox_area"] / (train_annot_df["image_width"] * train_annot_df["image_height"])
train_annot_df["foreground_ratio"].mean()

train_annot_df.groupby("category_name")["foreground_ratio"].mean()

#%%

train_annot_df.groupby("category_name")["relative_bbox_area"].mean()

#%%

train_annot_df.image_id.nunique()

#%%

train_annot_df.groupby("category_name")["image_id"].nunique().sum()
# %%

class ObjectStats:
    def __init__(self, coco_ann: Union[pd.DataFrame, str, Path], bins=None, n_bins=5, strategy="quantile"):
        if isinstance(coco_ann, str):
            coco_ann = coco_annotation_to_df(coco_ann)
        self.df = coco_ann.copy()
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
        self.df["center_x"] = (self.df["bbox_x"] + self.df["bbox_w"]) / 2
        self.df["center_y"] = (self.df["bbox_y"] + self.df["bbox_h"]) / 2

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
                                 
        objects_center_y_norm = {"mean": self.df.groupby("category_name")["center_y_norm"].mean().to_dict(),
                                "median": self.df.groupby("category_name")["center_y_norm"].median().to_dict(),
                                "std": self.df.groupby("category_name")["center_y_norm"].std().to_dict(),
                                "min": self.df.groupby("category_name")["center_y_norm"].min().to_dict(),
                                "max": self.df.groupby("category_name")["center_y_norm"].max().to_dict()
                                }
                        
        
        bbox_stats_area = {
                            "mean": self.df["bbox_area"].mean(),
                            "median": self.df["bbox_area"].median(),
                            "std": self.df["bbox_area"].std(),
                            "min": self.df["bbox_area"].min(),
                            "max": self.df["bbox_area"].max()
                        }
        bbox_stats_area_norm = {"mean": self.df["relative_bbox_area"].mean(),
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
        bbox_stats_center_y_norm =  {"mean": self.df["center_y_norm"].mean(),
                                    "median": self.df["center_y_norm"].median(),
                                    "std": self.df["center_y_norm"].std(),
                                    "min": self.df["center_y_norm"].min(),
                                    "max": self.df["center_y_norm"].max()
                                    }
        
        objects_stats = {"area": objects_area_stats,
                        "area_norm": objects_area_norm_stats,
                        "aspect_ratio": objects_aspect_ratio_stats,
                        "height": objects_height,
                        "width": objects_width,
                        "center_x": objects_center_x,
                        "center_y": objects_center_y,
                        "relative_x_center": objects_relative_x_center,
                        "center_y_norm": objects_center_y_norm
                        }
        bbox_stats = {"aspect_ratio": bbox_stats_aspect_ratio,
                       "area": bbox_stats_area,
                       "area_norm": bbox_stats_area_norm,
                        "height": bbox_stats_height,
                        "width": bbox_stats_width,
                        "center_x": bbox_stats_center_x,
                        "center_y": bbox_stats_center_y,
                        "relative_x_center": bbox_stats_relative_x_center,
                        "center_y_norm": bbox_stats_center_y_norm
                        }
        
        result = {"objects_stats": objects_stats,
                    "bbox_stats": bbox_stats
                }
        return result
    
    def spatial_distribution(self, bins=20):
        heatmap, xedges, yedges = np.histogram2d(self.df["relative_x_center"], 
                                                 self.df["center_y_norm"], 
                                                 bins=bins, range=[[0, 1], [0, 1]]
                                                 )
        res = {
            "heatmap": heatmap,
            "xedges": xedges,
            "yedges": yedges
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
        areas = self.df[field_name]#.clip(1e-9, 1.0)
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
        areas = self.df[field_name]#.clip(1e-9, 1.0)
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
    
    
    
# %%
#ann_df = coco_annotation_to_df(train_annfile)
train_objstats = ObjectStats(coco_ann=train_annfile, n_bins=5, strategy="quantile")
val_objstats = ObjectStats(coco_ann=val_annfile, n_bins=5, strategy="quantile")
test_objstats = ObjectStats(coco_ann=test_annfile, n_bins=5, strategy="quantile")


#%%

train_df = train_objstats.df
val_df = val_objstats.df
test_df = test_objstats.df

#%%

train_df.area_bin_label
# %%
train_summary = train_objstats.summary()
val_summary = val_objstats.summary()
test_summary = test_objstats.summary()
# %%
train_summary

#%%

train_df.area_bin.isna().sum()

#%%

train_df[train_df["foreground_ratio"] > 1]

#%%

train_summary["difficulty"]["bbox_stats"]["bbox_area_ratio_per_bin"]
# %%

@dataclass
class SplitInputStore:
    train: Union[str, Path]  # annotation file path
    val: Union[str, Path]
    test: Union[str, Path]  

@dataclass
class ImageStatsResult:
    ...

@dataclass
class ObjectStatsResult:
    ...
    
    
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
    
    def compute_drift(self, field_names = ["area_bin_label"]):
        if not hasattr(self, "split_stats"):
            self.split_stats = self.compute_stats()
            split_dfs = self.split_stats["split_dfs"]
        else:
            split_dfs = self.split_stats["split_dfs"]
        
        self.drift_results = {}
        split_pairs = list(combinations(split_dfs.keys(), 2))  
        
        for pair in split_pairs:
            s1, s2 = pair
            df1, df2 = split_dfs[s1], split_dfs[s2]
            for field_name in field_names:
                labels = sorted(set([df[field_name].dropna().unique() for nm, df in split_dfs.items()])) 
                kl = kl_divergence_between_distributions(df1, df2, field_name=field_name, labels=labels)
                js = js_divergence_between_distributions(df1, df2, field_name=field_name, labels=labels)
                self.drift_results[pair] = {"kl_divergence": kl, 
                                            "js_divergence": js,
                                            "metric": field_name
                                            }
        
        return self.drift_results
        
        

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    res = np.sum(p * np.log(p / q))  
    return res


def kl_divergence_between_distributions(df1, df2, field_name, labels):
    p = df1[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    q = df2[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    kl = kl_divergence(p, q)
    return kl


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    #js = 0.5 * (kl_pm + kl_qm)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js

def js_divergence_between_distributions(df1, df2, field_name, labels):
    p = df1[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)#.sort_index()
    q = df2[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)#.sorted_index()
    js = js_divergence(p, q)
    return js


def compute_js_divergence_per_object(df1, df2, field_name, labels,
                                     category_field="category_name"
                                     ):
    classes = sorted(set(df1[category_field]) | set(df2[category_field]))
    results = {}
    for cls in classes:
        p = (df1[df1[category_field] == cls][field_name]
             .value_counts(normalize=True)
             .reindex(labels, fill_value=0)#.sort_index()
             )
        q = (df2[df2[category_field] == cls][field_name]
             .value_counts(normalize=True)
             .reindex(labels, fill_value=0)#.sort_index()
             )
        js = js_divergence(p, q)
        results[cls] = js
    return results


#%%

areas = train_df["relative_bbox_area"]#.clip(1e-9, 1.0)

#%%

areas

bins = np.quantile(areas, np.linspace(0, 1, 5 + 1))
# %%
min_area = areas[areas > 0].min()
bins = np.logspace(np.log10(min_area), 0, 5 + 1)

bins = np.linspace(0, 1, 5 +1)
#%%

for i in bins:
    print(f"{i:.6f}")
    break


# %%
pd.cut(areas, bins=bins, include_lowest=True)#.value_counts().sort_index()
# %%
data_a = [i for i in np.arange(1, 100, 2)]
# %%
data_b = [i for i in np.arange(51, 150, 2)]
# %%
bins_exp = np.quantile(data_a, np.linspace(0, 1, 5 + 1))
# %%
pd.cut(data_a, bins=bins_exp, include_lowest=True)
# %%
pd.cut(data_b, bins=bins_exp, include_lowest=True)
# %%
eq_bins = np.linspace(np.min(data_a), np.max(data_a), 5 +1)
eq_bins = np.concatenate(([-np.inf], eq_bins, [np.inf]))

pd.cut(data_b, bins=eq_bins, include_lowest=True)
# %%
#bins = 

np.logspace(np.log10(np.min(data_a)), np.log10(np.max(data_a)), 5 + 1)
# %%

val_df["area_bin_label"].isna().sum()


#%%

train_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/train"
val_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/valid"
test_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/test"

obj_stats_kwargs = {"train": {"coco_ann": train_annfile, "n_bins": 5, "strategy": "quantile"},
                    "val": {"coco_ann": val_annfile, "n_bins": 5, "strategy": "quantile"},
                    "test": {"coco_ann": test_annfile, "n_bins": 5, "strategy": "quantile"}
                }

image_stats_kwargs = {"train": {"image_dir": train_imgdir},
                        "val": {"image_dir": val_imgdir},
                        "test": {"image_dir": test_imgdir}
                    }


#%%

print(f"obj_stats_kwargs: {obj_stats_kwargs}")

#%%
split_stats_cls = SplitStats(object_stats_cls=ObjectStats,
                            image_loader_cls=ImageBatchDataset, 
                            object_stats_kwargs=obj_stats_kwargs,
                            image_stats_kwargs=image_stats_kwargs
                            )  



#%%

split_stats_res = split_stats_cls.compute_stats()

#%%

split_stats_res["image_stats"].val

#%%

split_stats_res["split_dfs"]

#%%

split_stats_res["object_stats"].test


#%%

sorted(set([df["area_bin_label"].dropna().values for nm, df in split_stats_res["split_dfs"].items()]))

#%%

[df["area_bin_label"].dropna().values for nm, df in split_stats_res["split_dfs"].items()]
#%%

split_drift_res = split_stats_cls.compute_drift()


#%%
pd.concat([train_df["relative_bbox_area"], val_df["relative_bbox_area"]])


#%%

class DriftStats:
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
        
        """
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

#%%

train_df.columns 

val_df.columns   
#%%


drift_cls = DriftStats(reference_distribution=train_df,
                       comparison_distribution=val_df,
                       field_to_bin="relative_bbox_area",
                       name_bin_field_as="bbox_area_bin",
                       name_bin_field_label_as="bbox_area_bin_label"
                       )


#%%

train_val_drift = drift_cls.compute_drift()


#%%

train_val_drift

#%%

train_refdf = drift_cls.reference_distribution

train_refdf.groupby("bbox_area_bin_label").size()

#%%
val_compdf = drift_cls.comparison_distribution
val_compdf.groupby("bbox_area_bin_label").size()


#%%

train_test_drift_cls = DriftStats(reference_distribution=train_df,
                                 comparison_distribution=test_df,
                                 field_to_bin="relative_bbox_area",
                                name_bin_field_as="bbox_area_bin",
                                name_bin_field_label_as="bbox_area_bin_label"
                                 )

#%%

train_test_drift_res = train_test_drift_cls.compute_drift()


train_test_drift_res

#%%

test_compdf = train_test_drift_cls.comparison_distribution
test_compdf.groupby("bbox_area_bin_label").size()


#%%

class DomainShiftScore:
    def __init__(self, drift_scores: dict,
                 weights: dict = None,
                 normalize_drift_scores = False,
                 **kwargs
                 ):
        self.drift_scores = drift_scores
        self.weights = weights
        self.normalize = normalize_drift_scores
        
        
        if self.weights is None:
            total_num_drifts = len(self.drift_scores)
            self.weights = {k: 1.0/total_num_drifts for k in self.drift_scores.keys()}
        
        #if self.weights.values().sum() > 1.0:
        if normalize_drift_scores:
            max_value = max(self.drift_scores.values())
            self.norm_drift_scores = {k: v/max_value for k,v in self.drift_scores.items()}
        else:
            self.norm_drift_scores = drift_scores
            
    def compute_composite_score(self):
        self.composite_score =  sum(self.norm_drift_scores[k]*self.weights[k] 
                                    for k in self.norm_drift_scores
                                    )
        return self.composite_score
    
    def get_max_drift(self):
        return max(self.norm_drift_scores.values())
    
    def ranked(self):
        return sorted(self.norm_scores.items(), key=lambda x: x[1], reverse=True) 
    
    def explain(self): 
        lines = ["Domain Shift Breakdown:"] 
        for k, v in self.ranked(): 
            lines.append(f" - {k}: {v:.4f}") 
        lines.append(f"\nComposite Score: {self.composite():.4f}") 
        lines.append(f"Max Score: {self.max():.4f}") 
        return "\n".join(lines)
 
 
 
#%%

class DriftMetricSuite:
    def __init__(self, distributions: Dict[str, pd.DataFrame],
                metrics: Union[str, List[str]],
                field_to_bin: Union[str, List[str]],
                **kwargs
                ):
        
        self.distributions = distributions
        self.metrics = metrics
        self.distribution_pairs = list(combinations(distributions.keys(), 2))
        self.strategy = kwargs.get("strategy", "quantile")
        self.n_bins = kwargs.get("n_bins", 5)
        self.bins = kwargs.get("bins")
        self.include_overflow_bin = kwargs.get("include_overflow_bin", False)
        
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
            
    def drift_metrics(self):
        self.drift_results = {}
        for pair in self.distribution_pairs:
            ref, comp = pair
            ref_df = self.distributions[ref]
            comp_df = self.distributions[comp]
            
            for field in self.field_to_bin:
                name_bin_field_as = f"{field}_bin"
                name_bin_field_label_as = f"{name_bin_field_as}_label"
                for metric in self.metrics:
                    drift_cls = DriftStats(reference_distribution=ref_df,
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
                    
        return self.drift_results


def plot_drift_radar(drift_scores: List[float], 
                     drift_properties: List[str],
                     title=None 
                     ):
    fig = px.line_polar(r=drift_scores, theta=drift_properties,
                        template="plotly_dark",
                        title=title
                        )
    return fig
            
#%%
train_df.columns

distributions = {"train": train_df,
                 "val": val_df,
                 "test": test_df
                 }

metrics = ["kl", "js"]

field_to_bin = ['relative_bbox_area', 'bbox_aspect_ratio',
                'foreground_to_background_area_per_image', 
                'occupancy_per_image'
                ]

#%%

drift_suite_cls = DriftMetricSuite(distributions=distributions,
                                    metrics=metrics, field_to_bin=field_to_bin
                                    )


#%%
drift_results = drift_suite_cls.drift_metrics()

#%%

drift_results


#%%
drift_results.keys()
drift_radar_data = {}
drift_property = []
drift_metric = []
for k, v in drift_results.items():
    if k.startswith("('train', 'val')") and k.endswith("js"):
        drift_property.append(v["property"])
        drift_metric.append(v["js"])
    
#%%



#%%

drift_radar_plot = plot_drift_radar(drift_scores=drift_metric,
                                    drift_properties=drift_property,
                                    title="JS Divergence"
                                    )

#%%

drift_radar_plot



#%%
import pandas as pd
from datetime import datetime

def create_dataset_card(df: pd.DataFrame, output_path: str = "DATASET_CARD.md"):
    """
    Generate a dataset card summarizing dataset metadata and computed data-centric ML metrics.
    """

    # Basic dataset stats
    num_images = df["image_id"].nunique()
    num_annotations = len(df)
    classes = df["category_id"].nunique() if "category_id" in df.columns else "N/A"

    # Aggregate metrics (per-image)
    per_image = df.drop_duplicates("image_id")

    # Build Markdown content
    md = f"""
# 📘 Dataset Card

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Dataset Overview

- **Number of images:** {num_images}
- **Number of annotations:** {num_annotations}
- **Number of classes:** {classes}

This dataset card summarizes the dataset and the data-centric ML metrics computed so far.

---

## 2. Annotation Statistics

### 2.1 Bounding Box Metrics (Per-object)

| Metric | Description |
|--------|-------------|
| `relative_bbox_area` | Normalized bbox area relative to image area |
| `bbox_aspect_ratio` | Width / height |
| `relative_x_center`, `center_y_norm` | Normalized bbox center coordinates |

**Summary:**

- Mean bbox area norm: {df["relative_bbox_area"].mean():.4f}
- Median bbox area norm: {df["relative_bbox_area"].median():.4f}
- Mean aspect ratio: {df["bbox_aspect_ratio"].mean():.4f}

---

## 3. Per-Image Metrics

### 3.1 Object Count & Size Diversity

| Metric | Description |
|--------|-------------|
| `num_bboxes_per_image` | Number of objects per image |
| `relative_bbox_area_variance_per_image` | Variance of object sizes within each image |

**Summary:**

- Mean objects per image: {per_image["num_bboxes_per_image"].mean():.2f}
- Mean bbox size variance: {per_image["relative_bbox_area_variance_per_image"].mean():.4f}

---

### 3.2 Foreground & Background Structure

| Metric | Description |
|--------|-------------|
| `foreground_union_area_per_image` | True foreground area (union of all bboxes) |
| `occupancy_per_image` | Sum of bbox areas / image area |
| `background_area_norm` | Background area / image area |
| `foreground_to_background_area_per_image` | Foreground-background contrast (union-based) |
| `foreground_occupancy_to_background_occupancy` | Contrast using summed bbox areas |

**Summary:**

- Mean foreground ratio (union): {per_image["foreground_union_area_per_image"].mean():.4f}
- Mean occupancy (sum of areas): {per_image["occupancy_per_image"].mean():.4f}
- Mean foreground/background contrast (union): {per_image["foreground_to_background_area_per_image"].mean():.4f}

---

## 4. Data-Centric ML Interpretation

### 4.1 What These Metrics Reveal

- **Scale drift:** via `relative_bbox_area`, `avg_relative_bbox_area`, `relative_bbox_area_variance_per_image`
- **Composition drift:** via `num_bboxes_per_image`
- **Scene density drift:** via `occupancy_per_image`
- **Foreground-background structure:** via `foreground_union_area_per_image`, contrast ratios
- **Annotation style drift:** via differences between union-based and sum-based metrics

These metrics together provide a **non-redundant, multi-axis view** of dataset quality and domain shift.

---

## 5. Known Limitations & Future Work

- Add class distribution metrics  
- Add image-level visual statistics (brightness, contrast, entropy)  
- Add embedding-based drift metrics  
- Add spatial distribution heatmaps  
- Add temporal or source-based metadata if available  

---

## 6. Citation

If you use this dataset card or the metrics, please cite your project or organization accordingly.

---

*This dataset card was automatically generated as part of a data-centric ML pipeline.*
"""

    # Write to file
    with open(output_path, "w") as f:
        f.write(md)

    print(f"Dataset card created at: {output_path}")


#%%

create_dataset_card(train_df)
# %%

js_divergence_between_distributions(train_df, val_df, field_name="area_bin_label", 
                                    labels=train_objstats.area_bin_labels)

