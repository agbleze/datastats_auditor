
from ..io.baseio import BaseAnnotationDFImporter
import json
import numpy as np
import pandas as pd
from pandas import json_normalize


class CocoAnnDFImporter(BaseAnnotationDFImporter):
    name = "coco_ann_importer"
    status = "experimental"
    
    
    def __init__(self, coco_annotation_file):
        self.coco_annotation_file = coco_annotation_file
    
        
    def load(self):
        
        with open(self.coco_annotation_file, "r") as annot_file:
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
