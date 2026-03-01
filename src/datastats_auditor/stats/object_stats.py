

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
#from seaborn import heatmap
from shapely.geometry import box
from shapely.ops import unary_union
import plotly.express as px
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

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



def compute_spatial_drift(spatial_A, spatial_B,
                            xy_colname="heatmap",
                            x_colname="px",
                            y_colname="py",
                            ):
        H_A, H_B = spatial_A[xy_colname], spatial_B[xy_colname]
        px_A, px_B = spatial_A[x_colname], spatial_B[x_colname]
        py_A, py_B = spatial_A[y_colname], spatial_B[y_colname]

        # 2D JS divergence
        js_2d = jensenshannon(H_A.ravel(), H_B.ravel())**2

        # 1D JS on marginals
        js_x = jensenshannon(px_A, px_B)**2
        js_y = jensenshannon(py_A, py_B)**2

        # 1D Wasserstein
        bins_1d = np.linspace(0, 1, len(px_A))
        w1_x = wasserstein_distance(bins_1d, bins_1d, px_A, px_B)
        w1_y = wasserstein_distance(bins_1d, bins_1d, py_A, py_B)

        # 2D Wasserstein
        x_centers = 0.5 * (spatial_A["xedges"][:-1] + spatial_A["xedges"][1:])
        y_centers = 0.5 * (spatial_A["yedges"][:-1] + spatial_A["yedges"][1:])
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        support = np.stack([X.ravel(), Y.ravel()], axis=1)

        w1_2d = wasserstein_distance_nd(
            support,
            support,
            u_weights=H_A.ravel(),
            v_weights=H_B.ravel(),
        )

        combined = js_2d + 0.5*(js_x + js_y) + 0.5*(w1_x + w1_y) + w1_2d

        return {"js_2d": js_2d,
                "js_x": js_x,
                "js_y": js_y,
                "w1_x": w1_x,
                "w1_y": w1_y,
                "w1_2d": w1_2d,
                "combined_score": combined,
                }


def plot_spatial_heatmaps(spatial_dict_A, spatial_dict_B, names=("A", "B")):
    H_A = spatial_dict_A["heatmap"]
    H_B = spatial_dict_B["heatmap"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{names[0]} Heatmap", f"{names[1]} Heatmap"]
    )

    fig.add_trace(
        go.Heatmap(z=H_A, colorscale="Viridis"),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=H_B, colorscale="Viridis"),
        row=1, col=2
    )
    
    annotations = []
    for i in range(H_A.shape[0]): 
        for j in range(H_A.shape[1]): 
            annotations.append( dict( x=j, y=i, xref="x1", yref="y1", 
                                     text=f"{H_A[i, j]:.3f}", 
                                     showarrow=False, 
                                     font=dict( color="white" if H_A[i, j] > H_A.max()/2 else "black", 
                                               size=7
                                               ) 
                                     ) 
                               )
            
    for i in range(H_B.shape[0]): 
        for j in range(H_B.shape[1]): 
            annotations.append( dict( x=j, y=i, xref="x2", yref="y2", text=f"{H_B[i, j]:.3f}", 
                                     showarrow=False, 
                                     font=dict( color="white" if H_B[i, j] > H_B.max()/2 else "black", 
                                               size=7 
                                               ) 
                                     ) 
                               )

    fig.update_layout(
        height=400,
        width=900,
        coloraxis=dict(colorscale="Viridis"),
        showlegend=False,
        annotations=annotations,
        template="plotly_dark"
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1) 
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    return fig



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
        #print(f"spatial_A.keys(): {spatial_A.keys()}")
        return compute_spatial_drift(spatial_A, spatial_B,
                                    xy_colname=xy_colname,
                                    x_colname=x_colname,
                                    y_colname=y_colname
                                    )
    
    
    # def compute_spatial_drift(spatial_A, spatial_B,
    #                         xy_colname="heatmap",
    #                         x_colname="px",
    #                         y_colname="py",
    #                         ):
    #     H_A, H_B = spatial_A[xy_colname], spatial_B[xy_colname]
    #     px_A, px_B = spatial_A[x_colname], spatial_B[x_colname]
    #     py_A, py_B = spatial_A[y_colname], spatial_B[y_colname]

    #     # 2D JS divergence
    #     js_2d = jensenshannon(H_A.ravel(), H_B.ravel())**2

    #     # 1D JS on marginals
    #     js_x = jensenshannon(px_A, px_B)**2
    #     js_y = jensenshannon(py_A, py_B)**2

    #     # 1D Wasserstein
    #     bins_1d = np.linspace(0, 1, len(px_A))
    #     w1_x = wasserstein_distance(bins_1d, bins_1d, px_A, px_B)
    #     w1_y = wasserstein_distance(bins_1d, bins_1d, py_A, py_B)

    #     # 2D Wasserstein
    #     x_centers = 0.5 * (spatial_A["xedges"][:-1] + spatial_A["xedges"][1:])
    #     y_centers = 0.5 * (spatial_A["yedges"][:-1] + spatial_A["yedges"][1:])
    #     X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    #     support = np.stack([X.ravel(), Y.ravel()], axis=1)

    #     w1_2d = wasserstein_distance_nd(
    #         support,
    #         support,
    #         u_weights=H_A.ravel(),
    #         v_weights=H_B.ravel(),
    #     )

    #     combined = js_2d + 0.5*(js_x + js_y) + 0.5*(w1_x + w1_y) + w1_2d

    #     return {
    #             "js_2d": js_2d,
    #             "js_x": js_x,
    #             "js_y": js_y,
    #             "w1_x": w1_x,
    #             "w1_y": w1_y,
    #             "w1_2d": w1_2d,
    #             "combined_score": combined,
    #         }



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
 
 

class DriftMetricSuite:
    def __init__(self, distributions: Dict[str, pd.DataFrame],
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
            
    def drift_metrics(self):
        self.drift_results = {}
        self.spatial_drift_result = {}
        self.spatial_distribution = {}
        self.spatial_heatmap = {}
        
        for pair in self.distribution_pairs:
            ref, comp = pair
            ref_df = self.distributions[ref]
            comp_df = self.distributions[comp]
            
            if self.kwargs.get("compute_spatial_drift", False):
                x_coordinate_field = self.kwargs.get("x_coordinate_field")
                y_coordinate_field = self.kwargs.get("y_coordinate_field")
                
                spatial_drift = DriftStats(reference_distribution=ref_df,
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
                
                self.spatial_drift_result[pair] = DriftStats.get_spatial_drift(self.spatial_distribution[pair[0]],
                                                                                self.spatial_distribution[pair[1]],
                                                                                )
                
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
        #print(f"spatial_distr[pair[0]]: {spatial_distr[pair[0]]}")
        #print(f"spatial_distr[pair[1]]: {spatial_distr[pair[1]]}")
        # self.spatial_drift_result[pair] = DriftStats.get_spatial_drift(spatial_distr[pair[0]],
        #                                                             spatial_distr[pair[1]],
        #                                                             )  
        return {"drift": self.drift_results, 
                "spatial_drift": self.spatial_drift_result,
                "spatial_distribution": self.spatial_distribution,
                "spatial_heatmap": self.spatial_heatmap
                }    


def plot_drift_radar(drift_scores: List[float], 
                     drift_properties: List[str],
                     title=None 
                     ):
    fig = px.line_polar(r=drift_scores, theta=drift_properties,
                        template="plotly_dark",
                        title=title
                        )
    return fig
  
 

def compute_stats(df, prop, group="category_name",
                  stats=["mean", "std", "min", "max", "median"]
                  ):
    if group:
        df = df.groupby(group)
    return (df[prop]
          .agg(stats).reset_index()
            )




def plot_groupbar(summary_df, x="category_name", **kwargs):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_df[x],
        y=summary_df["max"].values,
        name='Max',
        marker_color='indianred',
        text=summary_df["max"].values,
        textposition='auto',
        texttemplate='%{text:.3f}',
        textangle=90,
        showlegend=kwargs.get("showlegend", False)
    ))
    fig.add_trace(go.Bar(
        x=summary_df[x],
        y=summary_df["min"].values,
        name='Min',
        marker_color='lightsalmon',
        text=summary_df["min"].values,
        texttemplate='%{text:.3f}',
        textposition='outside',
        textangle=90,
        showlegend=kwargs.get("showlegend", False),
    ))
    fig.add_trace(go.Bar(
        x=summary_df[x],
        y=summary_df["mean"].values,
        name='Mean',
        marker_color='mediumpurple',
        text=summary_df["mean"].values,
        texttemplate='%{text:.3f}', 
        textposition='outside',
        textangle=90,
        showlegend=kwargs.get("showlegend", False)
    ))
    fig.add_trace(go.Bar(
        x=summary_df[x], 
        y=summary_df["std"].values,
        name='STD',
        marker_color='midnightblue',
        text=summary_df["std"].values,
        textposition='outside',
        texttemplate='%{text:.3f}',
        textangle=90,
        showlegend=kwargs.get("showlegend", False)
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                    uniformtext=dict(mode="show", 
                                    minsize=5,
                                    ),
                    template="plotly_dark",
                    showlegend=kwargs.get("showlegend", False)
                    )
    return fig



def plot_bar(df: pd.DataFrame, 
             x="category_name", 
            y="count", 
            **kwargs
            ):
    """
    
    kwargs:
        title
        template
        color_discrete_sequence
        labels
        facet_row
        facet_col
        facet_row_spacing
        facet_col_spacing
        barmode
        height
        width
    """
    fig = px.bar(df, x=x, y=y,
                 title=kwargs.get("title"),
                 template=kwargs.get("template", "plotly_dark"), 
                 color=x,
                 color_discrete_sequence=kwargs.get("color_discrete_sequence", px.colors.qualitative.Bold),
                 labels=kwargs.get("labels", {x: "Category", y: "Count"}),
                 text=y,
                 facet_row=kwargs.get("facet_row"),
                 facet_col=kwargs.get("facet_col"),
                 facet_row_spacing=kwargs.get("facet_row_spacing", 0.02),
                 facet_col_spacing=kwargs.get("facet_col_spacing", 0.02),
                 barmode=kwargs.get("barmode", "relative"),
                 height=kwargs.get("height"),
                 width=kwargs.get("width")
                 )
    #fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def get_subplot_indices(nrows, ncols):
    return list(itertools.product(range(1, nrows+1),
                                  range(1, ncols+1)
                                  )
                )
    


def make_split_plot(splits: dict, **kwargs):
    """
    splits = {
        "train": train_plot,
        "val": val_plot,
        "test": test_plot
    }
    """
    num = len(splits)
    nrows = kwargs.get("rows", 1)
    ncols = kwargs.get("cols", num)
    
    subplot_indices = get_subplot_indices(nrows, ncols)
    
    fig = make_subplots(rows=nrows, cols=ncols, 
                        subplot_titles=[k.capitalize() for k in splits]
                        )

    col = 1
    for (row, col), (name, plot) in zip(subplot_indices, splits.items()):
        for trace in plot.data:
            fig.add_trace(trace, row=row, col=col)
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)
    fig.update_traces(textangle=kwargs.get("textangle", -90), 
                      cliponaxis=kwargs.get("cliponaxis", False)
                      )
    fig.update_layout(height=kwargs.get("height", 400), 
                      width=kwargs.get("width", 300*num), 
                      showlegend= kwargs.get("showlegend", False),
                      margin=kwargs.get("margin", dict(l=30, r=20, t=20)),
                      template=kwargs.get("template", "plotly_dark"),
                      uniformtext_minsize=kwargs.get("uniformtext_minsize", 10),
                      uniformtext_mode=kwargs.get("uniformtext_mode", "show")
                      )
    return fig

    

def compute_bins(reference_distribution,
                 comparison_distribution,
                 n_bins=None, field_to_bin=None, 
                 
                    strategy=None,
                    ):
        # if strategy is None:
        #     strategy = self.strategy
            
        # if field_to_bin is None:
        #     field_to_bin = self.field_to_bin
        # if n_bins is None:
        #     n_bins = self.n_bins
        #areas = self.df[field_name]#.clip(1e-9, 1.0)
        values = pd.concat([reference_distribution[field_to_bin], comparison_distribution[field_to_bin]])
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
        # if self.include_overflow_bin:
        #     bins = np.concatenate(([-np.inf], bins, [np.inf]))
        return bins



import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

def compute_spatial_distribution(df, 
                                 x_col="relative_x_center", 
                                 y_col="relative_y_center",
                                 **kwargs
                                 ):
    
    heatmap, xedges, yedges = np.histogram2d(df[x_col],
                                            df[y_col],
                                            bins=kwargs.get("bins", 10),
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


def compute_spatial_drift(spatial_A, spatial_B,
                          xy_colname="heatmap",
                          x_colname="px",
                          y_colname="py",
                          ):
    H_A, H_B = spatial_A[xy_colname], spatial_B[xy_colname]
    px_A, px_B = spatial_A[x_colname], spatial_B[x_colname]
    py_A, py_B = spatial_A[y_colname], spatial_B[y_colname]

    # 2D JS divergence
    js_2d = jensenshannon(H_A.ravel(), H_B.ravel())**2

    # 1D JS on marginals
    js_x = jensenshannon(px_A, px_B)**2
    js_y = jensenshannon(py_A, py_B)**2

    # 1D Wasserstein
    bins_1d = np.linspace(0, 1, len(px_A))
    w1_x = wasserstein_distance(bins_1d, bins_1d, px_A, px_B)
    w1_y = wasserstein_distance(bins_1d, bins_1d, py_A, py_B)

    # 2D Wasserstein
    x_centers = 0.5 * (spatial_A["xedges"][:-1] + spatial_A["xedges"][1:])
    y_centers = 0.5 * (spatial_A["yedges"][:-1] + spatial_A["yedges"][1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    support = np.stack([X.ravel(), Y.ravel()], axis=1)

    w1_2d = wasserstein_distance_nd(
        support,
        support,
        u_weights=H_A.ravel(),
        v_weights=H_B.ravel(),
    )

    combined = js_2d + 0.5*(js_x + js_y) + 0.5*(w1_x + w1_y) + w1_2d

    return {"js_2d": js_2d,
            "js_x": js_x,
            "js_y": js_y,
            "w1_x": w1_x,
            "w1_y": w1_y,
            "w1_2d": w1_2d,
            "combined_score": combined,
            }



def compute_quadrant_masses(heatmap):
    h = heatmap
    mid = h.shape[0] // 2

    Q1 = h[:mid, :mid].sum()
    Q2 = h[:mid, mid:].sum()
    Q3 = h[mid:, :mid].sum()
    Q4 = h[mid:, mid:].sum()

    return np.array([Q1, Q2, Q3, Q4])


def compute_quadrant_drift(spatial_A, spatial_B):
    qA = compute_quadrant_masses(spatial_A["heatmap"])
    qB = compute_quadrant_masses(spatial_B["heatmap"])

    # Normalize
    qA = qA / qA.sum()
    qB = qB / qB.sum()

    js_quad = jensenshannon(qA, qB)**2
    l1_quad = np.abs(qA - qB).sum()

    return {
        "quadrant_A": qA,
        "quadrant_B": qB,
        "js_quadrant": js_quad,
        "l1_quadrant": l1_quad,
    }





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
| `relative_x_center`, `relative_y_center` | Normalized bbox center coordinates |

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

if __name__ == "__main__":    
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
    train_annot_df["relative_y_center"] = train_annot_df["center_y"] / train_annot_df["image_height"]



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


    # %%

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


    #%%

    [df["area_bin_label"].dropna().values for nm, df in split_stats_res["split_dfs"].items()]
    #%%

    split_drift_res = split_stats_cls.compute_drift()


    #%%

    split_stats_res['split_dfs'].keys()
    #%%
    pd.concat([train_df["relative_bbox_area"], val_df["relative_bbox_area"]])


    #%%

    #%%

    train_df.columns 

    val_df.columns  

    #%%

    train_df["split_type"] = "train"
    val_df["split_type"] = "val"
    test_df["split_type"] = "test"


    #%%

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)


    #%%

    px.histogram(full_split_df, x="relative_bbox_area", 
                histnorm="probability",
                title="Distribution of Relative BBox Area by Split",
                    template="plotly_dark",
                    color="category_name",
                    facet_row="split_type",
                    facet_col_spacing=0.1,
                    height=800,
                    width=800,
                )
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

    #%%
            
    #%%
    train_df.columns

    #%%
    distributions = {"train": train_df,
                    "val": val_df,
                    "test": test_df
                    }

    metrics = ["kl", "js"]

    field_to_bin = ['relative_bbox_area', 'bbox_aspect_ratio',
                    'relative_x_center', 'relative_y_center',
                    'foreground_union_area_per_image',
                    'occupancy_per_image', 'background_area_per_image',
                    #'foreground_to_background_area_per_image', 
                    'background_area_norm',
                    #'foreground_occupancy_to_background_occupany', 
                    'num_bboxes_per_image',
                    'relative_bbox_area_variance_per_image'
                    ]

    #%%

    drift_suite_cls = DriftMetricSuite(distributions=distributions,
                                        metrics=metrics, field_to_bin=field_to_bin,
                                        #name_bin_field_as=None,
                                        #name_bin_field_label_as=None,
                                        compute_spatial_drift=True,
                                        x_coordinate_field="relative_x_center",
                                        y_coordinate_field="relative_y_center",
                                        spatial_strategy="equal",
                                        spatial_n_bins=10,
                                        )


    #%%
    drift_results = drift_suite_cls.drift_metrics()

    #%%

    drift_results["spatial_heatmap"][('train', 'val')]

    #%%

    drift_results["spatial_heatmap"][('train', 'test')]

    #%%

    drift_results["spatial_heatmap"][('val', 'test')]

    #%%

    drift_results["spatial_drift"]


    #%%

    train_grp_df = train_df.groupby("category_name")["occupancy_per_image"].agg(["mean", "std", "max", "min", "median"]).reset_index()


    #%%

    train_grp_df = compute_stats(train_df, prop="occupancy_per_image")#.columns
    val_grp_df = compute_stats(val_df, prop="occupancy_per_image")
    test_grp_df = compute_stats(test_df, prop="occupancy_per_image")
    
    #%%
    
    class ObjectSummaryStats:
        def __init__(self, property_names: Union[str,list],
                    split_dfs: dict[str, pd.DataFrame],
                    groupby_name="category_name",
                    stats = ["mean", "std", "min", "max", "median"]
                    ):
            if isinstance(property_names, str):
                property_names = [property_names]
            self.property_names = property_names
            self.split_dfs = split_dfs
            self.stats = stats
            self.groupby_name = groupby_name
            
        def compute_summary_stats(self):
            self.summary_results = {}
            
            for split_name, df in self.split_dfs.items():
                prop_results = {}
                for prop in self.property_names:
                    split_prop_stat = compute_stats(df=df, prop=prop, 
                                                    group=self.groupby_name,
                                                    stats=self.stats
                                                    )
                    prop_results[f"{prop}"] = split_prop_stat
                self.summary_results[split_name] = prop_results
            return self.summary_results
    
    #%%
    
    split_stats_res["split_dfs"]["train"].columns
    
    property_names = ['bbox_area', 'relative_bbox_area',
                      'bbox_aspect_ratio', 'center_x', 'center_y', 'relative_x_center',
                    'relative_y_center', 'foreground_union_area_per_image',
                    'occupancy_per_image', 'background_area_per_image',
                    'foreground_to_background_area_per_image', 
                    'background_area_norm',
                    'foreground_occupancy_to_background_occupany', 
                    'num_bboxes_per_image',
                    'relative_bbox_area_variance_per_image'
                    ]
    
    summary_stats_cls = ObjectSummaryStats(property_names=property_names, 
                                           split_dfs=split_stats_res["split_dfs"]
                                            )
    #%%
    
    summary_stat_res = summary_stats_cls.compute_summary_stats()
    
    #%%
    
    summary_stat_res["train"].keys()#["bbox_area"]
    
    #%%
    
    split_nm = list(summary_stat_res.keys())
    property_nms = list(summary_stat_res[split_nm[0]].keys())
    
    class SummaryPlot:
        def __init__(self, summary_stat_results: dict, **kwargs):
            self.summary_stat_results = summary_stat_results
            self.kwargs = kwargs
            
        def plot(self):
            split_nm = list(self.summary_stat_results.keys())
            property_nms = list(self.summary_stat_results[split_nm[0]].keys())
            
            self.prop_subplots = {}
            for prop in property_nms:
                split_plots = {}
                for index, spl in enumerate(split_nm):
                    df = self.summary_stat_results[spl][prop]
                    grpbar = plot_groupbar(summary_df=df, showlegend=index==0)
                    split_plots[f"{spl} {prop}"] = grpbar
                prop_subplot = make_split_plot(split_plots, rows=self.kwargs.get("rows", 3), 
                                               cols=self.kwargs.get("cols", 1),
                                                height=self.kwargs.get("height", 700),
                                                showlegend=True
                                                )
                self.prop_subplots[prop] = prop_subplot
                
            return self.prop_subplots
                    
            
    #%%
    
    summaryplot_cls = SummaryPlot(summary_stat_results=summary_stat_res)   
    
    #%%
    
    summary_subplots = summaryplot_cls.plot() 
    
    #%%
    summary_subplots['occupancy_per_image']#.keys()  
    
      
    #%%

    px.bar(train_grp_df, x="category_name", y='mean', barmode="group",
        color="")


    #%%


    import plotly.graph_objects as go
    
    #%%
    
    summary_stat_res["val"]["relative_bbox_area"]#.keys()

    #%%

    train_summary_grp_plot = plot_groupbar(train_grp_df, showlegend=True)
    val_summary_grp_plot = plot_groupbar(val_grp_df, showlegend=False)
    test_summary_grp_plot = plot_groupbar(test_grp_df, showlegend=False)
    #%%

    split_grpplot = {"train": train_summary_grp_plot,
                    "val": val_summary_grp_plot,
                    "test": test_summary_grp_plot
                    }


    #%%

    make_split_plot(split_grpplot, rows=3, cols=1,
                    height=700,
                    showlegend=True
                    )
    #%%
    drift_results.keys()
    drift_radar_data = {}
    drift_property = []
    drift_metric = []
    for k, v in drift_results["drift"].items():
        if k.startswith("('train', 'val')") and k.endswith("js"):
            drift_property.append(v["property"])
            drift_metric.append(v["js"])
        

    #%%

    drift_radar_plot = plot_drift_radar(drift_scores=drift_metric,
                                        drift_properties=drift_property,
                                        title="JS Divergence"
                                        )

    #%%

    drift_radar_plot

    #%%

    train_category_count_df = train_df.groupby("category_name").size().rename("count").reset_index()
    val_category_count_df = val_df.groupby("category_name").size().rename("count").reset_index()
    test_category_count_df = test_df.groupby("category_name").size().rename("count").reset_index()
    #%%

    train_obj_count_plot = px.bar(train_category_count_df, x="category_name", y="count",
                                title="Category Distribution in Train Set",
                                template="plotly_dark", color="category_name",
                                color_discrete_sequence=px.colors.qualitative.Plotly,
                                    labels={"category_name": "Category", "count": "Count"},
                                    text="count", 
                                )

    #%%



    train_obj_count_plot.update_layout(showlegend=False, xaxis_tickangle=-45)



    train_obj_count_plot = plot_bar(train_category_count_df, 
                                x="category_name", y="count",
                                    )
                                #title="Category Distribution in Train Set"

    val_obj_plot = plot_bar(val_category_count_df, 
                            x="category_name", y="count",
                            #title="Category Distribution in Val Set"
                            )
    test_obj_plot = plot_bar(test_category_count_df, 
                            x="category_name", y="count",
                            #title="Category Distribution in Test Set"
                            )


    #%%

    plot_bar(full_split_df.groupby(["split_type", "category_name"]).size().reset_index(name="count"),
            x="category_name", y="count",
            title="Category Distribution by Split",
            facet_row="split_type",
            color="split_type",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"category_name": "Category", "count": "Count", "split_type": "Split"},
            text="count",
            
            )#.update_layout(showlegend=False, xaxis_tickangle=-45)
    #%%

    from plotly.subplots import make_subplots


    #%%

    import itertools

    #%%

        
    # list(itertools.product(range(1, 2), range(1, 3), #repeat=2
    #                        )
    #      )


    #%%

    get_subplot_indices(1, 3)
    #%%

    #%%
    splits_obj_barplots = {
            "Train  Set": train_obj_count_plot,
            "Val  Set": val_obj_plot,
            "Test  Set": test_obj_plot
        }


    #%%

    split_bar_plot = make_split_plot(splits_obj_barplots, rows=3,
                                    cols=1
                                    )

    #%%

    split_bar_plot


    #%%

    # x (0.0078125, 0.9953125)
    # y (0.00859375, 0.98828125)
    #%%
    train_val_xbins = compute_bins(reference_distribution=train_df, 
                            comparison_distribution=val_df,
                            field_to_bin="relative_x_center", strategy="equal",
                            n_bins=10
                            )

    train_val_ybins = compute_bins(reference_distribution=train_df, 
                            comparison_distribution=val_df,
                            field_to_bin="relative_y_center", strategy="equal",
                            n_bins=10
                            )


    train_test_xbins = compute_bins(reference_distribution=train_df, 
                            comparison_distribution=test_df,
                            field_to_bin="relative_x_center", strategy="equal",
                            n_bins=10
                            )
    train_test_ybins = compute_bins(reference_distribution=train_df, 
                            comparison_distribution=test_df,
                            field_to_bin="relative_y_center", strategy="equal",
                            n_bins=10
                            )

    val_test_xbins = compute_bins(reference_distribution=val_df, 
                            comparison_distribution=test_df,
                            field_to_bin="relative_x_center", strategy="equal",
                            n_bins=10
                            )
    val_test_ybins = compute_bins(reference_distribution=val_df, 
                            comparison_distribution=test_df,
                            field_to_bin="relative_y_center", strategy="equal",
                            n_bins=10
                            )
    #%%

    import numpy as np 
    from scipy.spatial.distance import jensenshannon 
    from scipy.stats import wasserstein_distance, wasserstein_distance_nd

    #%%

    train_heatmap, train_xedges, train_yedges = np.histogram2d(train_df["relative_x_center"],
                                                                train_df["relative_y_center"],
                                                                #bins=10,
                                                                range=[[0, 1], [0, 1]],
                                                                bins=[train_val_xbins, train_val_ybins], 
                                                                )

    #%%

    train_heatmap #.shape

    #%%

    train_xedges.shape

    #%%

    train_yedges#.shape

    #%%
    train_heatmap_proba = train_heatmap / train_heatmap.sum()
    train_heatmap_proba#.sum()#.shape
    #%%

    (train_heatmap / train_heatmap.sum()).sum(axis=1)#.shape

    #%%
    train_df["relative_x_center"].min(), train_df["relative_x_center"].max()

    #%%
    train_df["relative_y_center"].min(), train_df["relative_y_center"].max()


    #%%

    val_heatmap, val_xedges, val_yedges = np.histogram2d(val_df["relative_x_center"],
                                            val_df["relative_y_center"],
                                            #bins=10,
                                            #range=[[0, 1], [0, 1]],    
                                        )

    #%%

    val_heatmap.ravel()

    val_heatmap_proba = val_heatmap / val_heatmap.sum()


    #%%

    jensenshannon(train_heatmap_proba.ravel(), val_heatmap_proba.ravel())**2

    #%%

    x_centers = (train_xedges[:-1] + train_xedges[1:]) * 0.5
    x_centers

    #%%

    y_centers = (train_yedges[:-1] + train_yedges[1:]) * 0.5
    y_centers

    #%%

    X, Y =np.meshgrid(x_centers, y_centers, indexing="ij")

    #%%

    support = np.stack([X.ravel(), Y.ravel()], axis=1)
    support.shape
    #%%

    wasserstein_distance_nd(train_xedges, train_yedges,
                            #support, support, 
                            u_weights=train_heatmap_proba.ravel(), 
                            v_weights=val_heatmap_proba.ravel()
                            )


    #%%


    #%%

    train_spatial = compute_spatial_distribution(train_df, 
                                                bins=[train_val_xbins, train_val_ybins]
                                                )
    val_spatial = compute_spatial_distribution(val_df, 
                                            bins=[train_val_xbins, train_val_ybins])
    test_spatial = compute_spatial_distribution(test_df, 
                                                bins=[train_test_xbins, train_test_ybins]
                                                )

    #%%
    train_spatial

    #%%
    train_val_spatial_drift = compute_spatial_drift(train_spatial, val_spatial)
    train_val_spatial_drift

    #%%

    train_test_spatial_drift = compute_spatial_drift(train_spatial, test_spatial)
    train_test_spatial_drift

    #%%

    val_test_spatial_drift = compute_spatial_drift(val_spatial, test_spatial)
    val_test_spatial_drift



    #%%

    #%%

    plot_spatial_heatmaps(train_spatial, val_spatial, names=("Train", "Val"))

    #%%

    plot_spatial_heatmaps(train_spatial, test_spatial, names=("Train Object centers distribution", "Test Object centers distribution"))

    #%%

    plot_spatial_heatmaps(val_spatial, test_spatial, names=("Val", "Test")) 
    #%%

    #%%

    train_val_quadrant_drift = compute_quadrant_drift(train_spatial, val_spatial)
    train_val_quadrant_drift

    #%%

    train_test_quadrant_drift = compute_quadrant_drift(train_spatial, test_spatial)
    train_test_quadrant_drift
    #%%

    #%%

    create_dataset_card(train_df)
    # %%

    js_divergence_between_distributions(train_df, val_df, field_name="area_bin_label", 
                                        labels=train_objstats.area_bin_labels)




    """

    # Bias and Unbalance Analysis
    Bar plots facet / subplot color by category_name to visualize class distribution differences across splits.

    object count distribution by split
    counts
    

    Histogram stack plot of object-level metrics (e.g., bbox area, aspect ratio) by split to visualize distributional differences.
    Split specific distribution 
    color=category_name
    barmode="stack"
    probability | count
    **** per object per split
    --- 'bbox_area', 'relative_bbox_area',
        'bbox_aspect_ratio', 
        
    
    *** per split    
    'foreground_union_area_per_image',
        'occupancy_per_image', 'background_area_per_image',
        'foreground_to_background_area_per_image', 'background_area_norm',
        'foreground_occupancy_to_background_occupany', 'num_bboxes_per_image',
        'relative_bbox_area_variance_per_image',
        

    Spatial distribution heatmaps of object locations (e.g., center_x, center_y) by split to visualize spatial biases or shifts.
        
    Histogram 2d 
    -- 'relative_x_center',
        'relative_y_center'
        
        
        
    Drift metrics (e.g., KL divergence, JS divergence) computed on binned versions of object-level metrics to quantify distributional shifts across splits.
    Radar plots of drift metrics by property to visualize which properties exhibit the most shift.

    Comaprison on Train vs Val, Train vs Test, Val vs Test
    --  'bbox_area', 'relative_bbox_area',
        'bbox_aspect_ratio', #'center_x', 'center_y', 
        'relative_x_center',
        'relative_y_center', 'foreground_union_area_per_image',
        'occupancy_per_image', 'background_area_per_image',
        'foreground_to_background_area_per_image', 'background_area_norm',
        'foreground_occupancy_to_background_occupany', 'num_bboxes_per_image',
        'relative_bbox_area_variance_per_image',






    Index(['id_annotation', 'image_id', 'category_id', 'bbox', 'area',
        'segmentation', 'iscrowd', 'file_name', 'image_height', 'image_width',
        'category_name', 'supercategory', 'bbox_x', 'bbox_y', 'bbox_w',
        'bbox_h', 'image_area', 
        
        'bbox_area', 'relative_bbox_area',
        'bbox_aspect_ratio', 'center_x', 'center_y', 'relative_x_center',
        'relative_y_center', 'foreground_union_area_per_image',
        'occupancy_per_image', 'background_area_per_image',
        'foreground_to_background_area_per_image', 'background_area_norm',
        'foreground_occupancy_to_background_occupany', 'num_bboxes_per_image',
        'relative_bbox_area_variance_per_image', 'area_bin', 'area_bin_label',
        'bbox_area_bin', 'bbox_area_bin_label', 'relative_bbox_area_bin',
        'relative_bbox_area_bin_label', 'bbox_aspect_ratio_bin',
        'bbox_aspect_ratio_bin_label', 'relative_x_center_bin',
        'relative_x_center_bin_label', 'relative_y_center_bin',
        'relative_y_center_bin_label', 'foreground_union_area_per_image_bin',
        'foreground_union_area_per_image_bin_label', 'occupancy_per_image_bin',
        'occupancy_per_image_bin_label', 'background_area_per_image_bin',
        'background_area_per_image_bin_label', 'background_area_norm_bin',
        'background_area_norm_bin_label', 'num_bboxes_per_image_bin',
        'num_bboxes_per_image_bin_label',
        'relative_bbox_area_variance_per_image_bin',
        'relative_bbox_area_variance_per_image_bin_label', 'split_type'],
        dtype='object')

    """