PDF_PATH = "datacard.pdf"

AUTHORSHIP = {"header": "Authorship & Ownership",
            "Organization": "ORG",
            "Industry": "TECH",
            "Dataset owners": "AI TEAM"                   
            }

DATA_COLLECTION = {"header": "Data Collection",
                    "Collection strategy": "camera recording",
                    "Devices": "mobile phone",
                    "Collection Sites": "farm",
                    "Environmental conditions": "indoor/outdoor, lighting, weather"
                    }


LABELLING = {"header": "Annotation & Labeling",
                    "Labeling method": "human",
                    "Label types": "bbox, segmentation mask",
                    "Annotation format": "COCO",
                    "Annotation review method": "(SME, same labelers, etc.)",
                    "Platform": "CVAT"    
                    }

LICENCE = {"header": "Licensing & Usage",
                "License": "BSD",
                "Usage restrictions": "Internal-only",
                "Redistribution policy": "N/A"
                }

TRANSFORMATION = {"header":"Transformation",
                "Technique":"Augmentation",
                "Parameters": {},
                "Libraries used": "Augment" 
                }


METRICS = ["kl", "js"]

FIELD_TO_BIN = ['relative_bbox_area', 'bbox_aspect_ratio',
                'relative_x_center', 'relative_y_center',
                'foreground_union_area_per_image',
                'occupancy_per_image', 'background_area_per_image',
                #'foreground_to_background_area_per_image', 
                'background_area_norm',
                #'foreground_occupancy_to_background_occupany', 
                'num_bboxes_per_image',
                'relative_bbox_area_variance_per_image'
                ]

INTENDED_TASKS = ["Object detection", "Fairness evaluation"]
PDF_PATH = "dataset_card.pdf"