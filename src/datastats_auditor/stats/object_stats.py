

#%%

import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from pandas import json_normalize
import json
from dataclasses import dataclass
from typings import Dict, Union, List, Optional
from datastats_auditor.src.datastats_auditor.stats.image_stats import ImageBatchDataset
from datastats_auditor.stats.image_stats import compute_dataset_stats, estimate_image_memory_size_GB, get_memory_info

#%%
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


#%%    

train_annfile = "/home/lin/codebase/chili_seg/Chili_seg.v3i.coco-segmentation/train/_annotations.coco.json"

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
train_annot_df["bbox_area_norm"] = train_annot_df["bbox_area"] / (train_annot_df["image_width"] * train_annot_df["image_height"])
train_annot_df["bbox_aspect_ratio"] = train_annot_df["bbox_w"] / train_annot_df["bbox_h"]


# %% #### Spatial Distribution Analysis ######
# compute object center coordinates
train_annot_df["center_x"] = (train_annot_df["bbox_x"] + train_annot_df["bbox_w"]) / 2
train_annot_df["center_y"] = (train_annot_df["bbox_y"] + train_annot_df["bbox_h"]) / 2

# normalize
train_annot_df["center_x_norm"] = train_annot_df["center_x"] / train_annot_df["image_width"]
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

small_objects = train_annot_df[train_annot_df["bbox_area_norm"] < 0.01]
small_ratio = len(small_objects) / len(train_annot_df)
small_ratio

clutter_score = objects_per_image.mean() / (train_annot_df["image_width"] * train_annot_df["image_height"]).mean()
clutter_score

train_annot_df["foreground_ratio"] = train_annot_df["bbox_area"] / (train_annot_df["image_width"] * train_annot_df["image_height"])
train_annot_df["foreground_ratio"].mean()

train_annot_df.groupby("category_name")["foreground_ratio"].mean()

#%%

train_annot_df.groupby("category_name")["bbox_area_norm"].mean()

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
            bins = self.compute_area_bins(n_bins=n_bins, strategy=strategy)
        self.bins = bins
        self.area_bin_labels = [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(len(bins)-1)]
        self.df = self.assign_area_bins(bins, self.area_bin_labels)           
        
    def _prepare(self):
        self.df.dropna(inplace=True)
        self.df["bbox_x"] = self.df["bbox"].apply(lambda b: b[0])
        self.df["bbox_y"] = self.df["bbox"].apply(lambda b: b[1])
        self.df["bbox_w"] = self.df["bbox"].apply(lambda b: b[2])
        self.df["bbox_h"] = self.df["bbox"].apply(lambda b: b[3])

        self.df["bbox_area"] = self.df["bbox_w"] * self.df["bbox_h"]
        self.df["bbox_area_norm"] = self.df["bbox_area"] / (self.df["image_width"] * self.df["image_height"])
        self.df["bbox_aspect_ratio"] = self.df["bbox_w"] / self.df["bbox_h"]

        # compute object center coordinates
        self.df["center_x"] = (self.df["bbox_x"] + self.df["bbox_w"]) / 2
        self.df["center_y"] = (self.df["bbox_y"] + self.df["bbox_h"]) / 2

        # normalize
        self.df["center_x_norm"] = self.df["center_x"] / self.df["image_width"]
        self.df["center_y_norm"] = self.df["center_y"] / self.df["image_height"]
        
        self.df["foreground_ratio"] = self.df["bbox_area"] / (self.df["image_width"] * self.df["image_height"])               

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
                              
        objects_area_norm_stats = {"mean": self.df.groupby("category_name")["bbox_area_norm"].mean().to_dict(),
                                    "median": self.df.groupby("category_name")["bbox_area_norm"].median().to_dict(),
                                    "std": self.df.groupby("category_name")["bbox_area_norm"].std().to_dict(),
                                    "min": self.df.groupby("category_name")["bbox_area_norm"].min().to_dict(),
                                    "max": self.df.groupby("category_name")["bbox_area_norm"].max().to_dict()
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
                            
        objects_center_x_norm = {"mean": self.df.groupby("category_name")["center_x_norm"].mean().to_dict(),
                                "median": self.df.groupby("category_name")["center_x_norm"].median().to_dict(),
                                "std": self.df.groupby("category_name")["center_x_norm"].std().to_dict(),
                                "min": self.df.groupby("category_name")["center_x_norm"].min().to_dict(),
                                "max": self.df.groupby("category_name")["center_x_norm"].max().to_dict()
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
        bbox_stats_area_norm = {"mean": self.df["bbox_area_norm"].mean(),
                                "median": self.df["bbox_area_norm"].median(),
                                "std": self.df["bbox_area_norm"].std(),
                                "min": self.df["bbox_area_norm"].min(),
                                "max": self.df["bbox_area_norm"].max()
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
        bbox_stats_center_x_norm  =   {"mean": self.df["center_x_norm"].mean(),
                                        "median": self.df["center_x_norm"].median(),
                                        "std": self.df["center_x_norm"].std(),
                                        "min": self.df["center_x_norm"].min(),
                                        "max": self.df["center_x_norm"].max()
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
                        "center_x_norm": objects_center_x_norm,
                        "center_y_norm": objects_center_y_norm
                        }
        bbox_stats = {"aspect_ratio": bbox_stats_aspect_ratio,
                       "area": bbox_stats_area,
                       "area_norm": bbox_stats_area_norm,
                        "height": bbox_stats_height,
                        "width": bbox_stats_width,
                        "center_x": bbox_stats_center_x,
                        "center_y": bbox_stats_center_y,
                        "center_x_norm": bbox_stats_center_x_norm,
                        "center_y_norm": bbox_stats_center_y_norm
                        }
        
        result = {"objects_stats": objects_stats,
                    "bbox_stats": bbox_stats
                }
        return result
    
    def spatial_distribution(self, bins=20):
        heatmap, xedges, yedges = np.histogram2d(self.df["center_x_norm"], 
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

        small_objects = self.df[self.df["bbox_area_norm"] <= small_object_threshold]
        small_ratio = len(small_objects) / len(self.df)
        large_objects = self.df[self.df["bbox_area_norm"] >= large_object_threshold]
        large_ratio = len(large_objects) / len(self.df)
        medium_objects = self.df[(self.df["bbox_area_norm"] > small_object_threshold) & (self.df["bbox_area_norm"] < large_object_threshold)]
        medium_ratio = len(medium_objects) / len(self.df)

        clutter_score = objects_per_image.mean() / (self.df["image_width"] * self.df["image_height"]).mean()

        foreground_ratio_mean = self.df["foreground_ratio"].mean()
        foreground_ratio_min = self.df["foreground_ratio"].min()
        foreground_ratio_max = self.df["foreground_ratio"].max()
        foreground_ratio_median = self.df["foreground_ratio"].median()
        foreground_ratio_std = self.df["foreground_ratio"].std()


        object_foreground_ratio_mean = self.df.groupby("category_name")["foreground_ratio"].mean().to_dict()
        object_foreground_ratio_max = self.df.groupby("category_name")["foreground_ratio"].max().to_dict()
        object_foreground_ratio_min = self.df.groupby("category_name")["foreground_ratio"].min().to_dict()
        object_foreground_ratio_median = self.df.groupby("category_name")["foreground_ratio"].median().to_dict()
        object_foreground_ration_std = self.df.groupby("category_name")["foreground_ratio"].std().to_dict()

        bbox_area_bins_ratio = self.df["area_bin_label"].value_counts(normalize=True).to_dict()
        object_bbox_area_per_bins = self.df.groupby(["area_bin_label", "category_name"]).size().unstack(fill_value=0)
        
        bbox_stats = {"objects_in_image": {"mean": avg_objects,
                                            "min": min_object_per_image,
                                            "max": max_object_per_image,
                                            "median": median_objects_per_image,
                                            "std": objects_per_image.std()
                                            },
                      "foreground_ratio": {"mean": foreground_ratio_mean,
                                            "min": foreground_ratio_min,
                                            "max": foreground_ratio_max,
                                            "median": foreground_ratio_median,
                                            "std": foreground_ratio_std
                                            },
                      "small_object": {"ratio": small_ratio},
                      "medium_object": {"ratio": medium_ratio},
                      "large_object": {"ratio": large_ratio},
                      "bbox_area_ratio_per_bin": bbox_area_bins_ratio,
                      "clutter_score": clutter_score,                      
                      }
        
        object_stats = {"images_per_object_ratio": images_per_object_ratio.to_dict(),
                        "foreground_ratio": {"mean": object_foreground_ratio_mean,
                                            "min": object_foreground_ratio_min,
                                            "max": object_foreground_ratio_max,
                                            "median": object_foreground_ratio_median,
                                            "std": object_foreground_ration_std
                                            },
                        "bbox_area_ratio_per_bin": object_bbox_area_per_bins.to_dict()
                        }
        
        self.difficulty_metrics ={"bbox_stats": bbox_stats, "object_stats": object_stats}
        
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
                                 field_name="bbox_area_norm", 
                                 strategy="quantile"):
        areas = self.df[field_name].clip(1e-9, 1.0)
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
    
    def compute_area_bins(self, n_bins=5, field_name="bbox_area_norm", 
                          strategy="quantile"
                          ):
        areas = self.df[field_name].clip(1e-9, 1.0)
        if strategy == "quantile":
            bins = np.quantile(areas, np.linspace(0, 1, n_bins + 1))
        elif strategy == "equal":
            bins = np.linspace(0, 1, n_bins +1)
        elif strategy == "log":
            min_area = areas[areas > 0].min()
            bins = np.logspace(np.log10(min_area), 0, n_bins + 1)
        else:
            raise ValueError(f"strategy must be 'quantile', 'equal', or 'log' and not {strategy}")
        return bins
    
    def assign_area_bins(self, bins, labels, 
                         field_to_bin="bbox_area_norm",
                         name_bin_field_as="area_bin", 
                         name_bin_field_label_as="area_bin_label"
                         ):
        self.df[name_bin_field_as] = pd.cut(self.df[field_to_bin], bins=bins, include_lowest=True)
        self.df[name_bin_field_label_as] = pd.cut(self.df[field_to_bin], bins=bins, labels=labels, include_lowest=True)
        return self.df
    
    
    
# %%
ann_df = coco_annotation_to_df(train_annfile)
objstats = ObjectStats(ann_df=ann_df, n_bins=5, strategy="equal")


#%%

df = objstats.df


#%%

df.area_bin_label
# %%
summary = objstats.summary()
# %%
summary
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
    def __init__(self, object_stats: ObjectStats,
                 image_stats: ImageBatchDataset,
                 object_stats_kwargs: Optional[Dict] = None,
                 image_stats_kwargs: Optional[Dict] = None,
                 **kwargs
                 ):
        self.object_stats = object_stats
        self.image_stats = image_stats
        self.object_stats_kwargs = object_stats_kwargs or {}
        self.image_stats_kwargs = image_stats_kwargs or {}
        
        self.imagestat_results = ImageStatsResult()
        self.objectstat_results = ObjectStatsResult()

    def compute_stats(self): 
        for split_nm, split_param in self.image_stats_kwargs.items():
            dataset = ImageBatchDataset(**split_param)
            imagestat_results = compute_dataset_stats(dataset)
            setattr(self.imagestat_results, split_nm, imagestat_results)

        split_df_collection = {}
        for split_nm, split_param in self.object_stats_kwargs.items():
            ann_df = coco_annotation_to_df(split_param["ann_file"])
            objstats = ObjectStats(ann_df=ann_df, **split_param)
            objstats_summary = objstats.summary()
            setattr(self.objectstat_results, split_nm, objstats_summary)
            split_df_collection[split_nm] = objstats.df    
        
        self.object_stats_results = self.objectstat_results
        self.image_stats_results = self.imagestat_results
        self.split_stats = {"image_stats": self.image_stats_results,
                            "object_stats": self.object_stats_results
                            }
        return self.split_stats
        
        
        

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
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
    p = np.asarray(p, dytpe=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    #js = 0.5 * (kl_pm + kl_qm)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js

def js_divergence_between_distributions(df1, df2, field_name, labels):
    p = df1[field_name].value_counts(normalize=True).reindex(labels, fill_value=0).sort_index()
    q = df2[field_name].value_counts(normalize=True).reindex(labels, fill_value=0).sorted_index()
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
             .reindex(labels, fill_value=0).sort_index()
             )
        q = (df2[df2[category_field] == cls][field_name]
             .value_counts(normalize=True)
             .reindex(labels, fill_value=0).sort_index()
             )
        js = js_divergence(p, q)
        results[cls] = js
    return results





