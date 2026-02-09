

#%%

import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from pandas import json_normalize
import json


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

images_per_object = train_annot_df.groupby("category_name").size()
images_per_object_ratio = images_per_object / images_per_object.sum()

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


# %%

class ObjectStats:
    def __init__(self, ann_df):
        self.df = ann_df.copy()
        self._prepare()
        
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
                                    "mean": self.df["bbox_aspect_ratio"].mean(),
                                    "median": self.df["bbox_aspect_ratio"].median(),
                                    "std": self.df["bbox_aspect_ratio"].std(),
                                    "min": self.df["bbox_aspect_ratio"].min(),
                                    "max": self.df["bbox_aspect_ratio"].max()
                                }
        bbox_stats_height = {"mean": self.df["bbox_h"].mean(),
                            "mean": self.df["bbox_h"].mean(),
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
                        "center_y": {
                                        "mean": self.df["center_y"].mean(),
                                        "median": self.df["center_y"].median(),
                                        "std": self.df["center_y"].std(),
                                        "min": self.df["center_y"].min(),
                                        "max": self.df["center_y"].max()
                                    },
                        "center_x_norm": {
                                        "mean": self.df["center_x_norm"].mean(),
                                        "median": self.df["center_x_norm"].median(),
                                        "std": self.df["center_x_norm"].std(),
                                        "min": self.df["center_x_norm"].min(),
                                        "max": self.df["center_x_norm"].max()
                                    },
                        "center_y_norm": {
                                        "mean": self.df["center_y_norm"].mean(),
                                        "median": self.df["center_y_norm"].median(),
                                        "std": self.df["center_y_norm"].std(),
                                        "min": self.df["center_y_norm"].min(),
                                        "max": self.df["center_y_norm"].max()
                                    }
                    }    
        
        {"objects_area_stats": objects_area_stats,
         "objects_area_norm_stats": objects_area_norm_stats,
         "objects_aspect_ratio_stats": objects_aspect_ratio_stats,
         "bbox_stats": bbox_stats
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
         "bbox_stats": {}
        }
    