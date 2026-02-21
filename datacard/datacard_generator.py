

#%%
from pathlib import Path

class DatasetCardCreator:
    def __init__(self, sections: dict, renderer):
        self.sections = sections
        self.renderer = renderer

    def generate(self, output_path: str = "DATASET_CARD.md"):
        text = self.renderer.render(self.sections)
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"Dataset card created at: {output_path}")
        return text
        
        




def build_motivations_and_use(
    purposes=None,
    domain_applications=None,
    problem_space=None,
    primary_motivations=None,
    intended_tasks=None,
    intended_objects=None,
):
    """
    Dynamically generates the 'Motivations & Use' section of the dataset card.
    """

    purposes = purposes or ["Training", "Validation", "Research", "Evaluation"]
    domain_applications = domain_applications or ["Machine Learning", "Computer Vision", "Object Recognition"]
    problem_space = problem_space or (
        "This dataset was created to support research, development, and experimentation "
        "in real-world industrial settings with constraints such as camera angle, orientation, "
        "lighting conditions, and environmental variations that are difficult to replicate "
        "in synthetic simulations."
    )
    primary_motivations = primary_motivations or (
        "Capture realistic and representative variations in annotations, foreground/background "
        "composition, and scene interactions within the target environments."
    )
    intended_tasks = intended_tasks or ["Object detection", "Person detection"]
    intended_objects = intended_objects or []

    # Format lists into bullet points
    def bullets(items):
        return "\n".join([f"- {item}" for item in items])

    section = f"""
# Motivations & Use

## Dataset Purpose(s)
{bullets(purposes)}

## Key Domain Application(s)
{bullets(domain_applications)}

## Problem Space
{problem_space}

## Primary Motivation(s)
{primary_motivations}

## Intended Use Case(s)

### Tasks
{bullets(intended_tasks)}

### Objects Labelled
{bullets(intended_objects) if intended_objects else "- (No specific objects selected)"}
"""
    return section.strip()        



#%%

class MarkdownRenderer:
    def render(self, sections: dict) -> str:
        md = "# 📘 Dataset Card\n\n"
        for title, content in sections.items():
            if content is None or not str(content).strip():
                continue
            md += f"## {title}\n\n{content.strip()}\n\n"
        return md
    
#%%

moti_content = build_motivations_and_use(
    intended_tasks=["Object detection", "Fairness evaluation"],
    intended_objects=["Person", "Helmet", "Safety vest"],
)
print(moti_content)
# %%
motsect = {"Motivation and Intended Use": moti_content}

mkd_render = MarkdownRenderer()

cardgen = DatasetCardCreator(sections=motsect, renderer=mkd_render)
# %%
cardgen.generate()
# %%
import pypandoc

#pypandoc.download_pandoc()
pypandoc.convert_file(
    "DATASET_CARD.md",
    "pdf",
    outputfile="DATASET_CARD.pdf"
)

# %%
def create_section(*args, **kwargs):
    kwargs = kwargs.get("kwargs")
    header = kwargs.get("header", "")
    header = f"📘 {header}" if header else ""
    sect_content = "\n".join([f"- {key}: {value}" for key, value in kwargs.items() if key != "header"])
    #sect_content = "\n".join(sect_content)
    # font-size:20px;
    section = f""" 
<H1 style="color:#2A7FFF; font-weight:bold;">
 {header}
</H1>

{sect_content}
    
    """
    return section.strip()
    
# %%
import uuid


summary_kwargs = {#"header": "1. Summary",
                "Name": "demo data",
                "Version ID": str(uuid.uuid1()),
                "Modality": "Image",
                "Number of images": 1234,
                "Number of objects": 5678,
                "Objects labeled": ["cocoa", "tomato", "coconut"],
                "Object counts per split": {"train": 123, "val": 23, "test": 56}
                }


auth_kwargs = {"header": "2. Authorship & Ownership",
                "Organization": "ORG",
                "Industry": "TECH",
                "Dataset owners": "AI TEAM"                   
                }

data_col_kwargs = {"header": "4. Data Collection",
                    "Collection strategy": "camera recording",
                    "Devices": "mobile phone",
                    "Collection Sites": "farm",
                    "Environmental conditions": "indoor/outdoor, lighting, weather"
                    }


labelling_kwargs = {"header": "5. Annotation & Labeling",
                    "Labeling method": "human",
                    "Label types": "bbox, segmentation mask",
                    "Annotation format": "COCO",
                    "Annotation review method": "(SME, same labelers, etc.)",
                    "Platform": "CVAT"    
                    }

licence_kwargs = {"header": "9. Licensing & Usage",
                "License": "BSD",
                "Usage restrictions": "Internal-only",
                "Redistribution policy": "N/A"
                }

trans_kwargs = {"header":"Transformation",
                "Technique":"Augmentation",
                "Parameters": {},

                "Libraries used": "Augment" 
    
}


split_kwargs = {"header": "Data Split Composition",
                "Algorithm": "similarity‑based splitter",
                    "Parameters": {"seed":123, "object_based": True, 
                                "train_ratio": 0.8
                                },
                    "package": "cluster-aware-spliter"
                }
#%%

summary_section = create_section(kwargs=summary_kwargs)
authorship_section = create_section(kwargs=auth_kwargs)
data_collection_section = create_section(kwargs=data_col_kwargs)
labelling_section = create_section(kwargs=labelling_kwargs)
split_section = create_section(kwargs=split_kwargs)
transformation_section = create_section(kwargs=trans_kwargs)
license_section = create_section(kwargs=licence_kwargs)

# %%
sections = {"Authorship & Ownership": authorship_section,
            "Data Summary": summary_section,            
            "Motivation and Intended Use": moti_content,
            "Data Collection": data_collection_section,
            "Annotation & Labeling": labelling_section,
            "Data Split Composition": split_section,
            "Transformation": transformation_section,
            "License & Usage": license_section
                       
            }
cardgen = DatasetCardCreator(sections=sections, renderer=mkd_render)
# %%
cardgen.generate()
# %%





















#%%
"""
TODO. CREate func to generate sections for this with each 
preprocessing step being a subheading to - Preprocessing & Data Preparation
6. Preprocessing & Data Preparation
This is where your detailed technical pipeline belongs.

6.1 Data Cleaning & Sampling
Deduplication (exact)

Algorithm

Parameters

Package + version

Deduplication (near‑duplicate)

Algorithm

Parameters

Package + version

Cross‑filtering

Algorithm

Parameters

Package + version



# TODO: Explore option of creating a decorator that can be used to
generate this section of card

7. Data Metrics (Data‑Centric ML Quality Metrics)
This is your signature section.

7.1 Bias & Fairness Metrics
(If applicable)

7.2 Structural Characteristics Metrics
(e.g., occupancy, foreground/background ratios)

7.3 Spatial Distribution Metrics
(e.g., bbox center heatmaps, spatial entropy)

7.4 Image‑Level Statistics
(brightness, contrast, entropy)

7.5 Object‑Level Statistics
(bbox area norm, aspect ratio, etc.)

7.6 Split‑Level Statistics
(per‑split distributions)

7.7 Split Drift Metrics
(JS divergence, histograms, plots)

This section is now cleanly grouped and expandable.
"""
    
    