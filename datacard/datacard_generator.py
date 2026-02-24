

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
from markdown_it import MarkdownIt
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

#%%  ###############

import plotly.express as px
import pandas as pd
from markdown import markdown
from weasyprint import HTML
import base64
from io import BytesIO

def generate_dataset_card_with_embedded_plot(
    df: pd.DataFrame,
    md_path="dataset_card.md",
    pdf_path="dataset_card.pdf"
):
    # ---------------------------------------------------------
    # 1. Create Plotly figure
    # ---------------------------------------------------------
    fig = px.histogram(
        df,
        x="bbox_area_norm",
        title="BBox Area Distribution",
        nbins=20
    )

    # ---------------------------------------------------------
    # 2. Convert Plotly figure → base64 PNG (no file saved)
    # ---------------------------------------------------------
    img_bytes = fig.to_image(format="png")  # requires kaleido
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    img_md = f"![plot](data:image/png;base64,{img_b64})"

    # ---------------------------------------------------------
    # 3. Build Markdown content with embedded image
    # ---------------------------------------------------------
    md_content = f"""
# Dataset Card

## Data-Centric Metrics

### BBox Area Distribution

The histogram below shows the distribution of normalized bounding box areas.

{img_md}


![alt-text-1]({img_md} "title-1") ![alt-text-2]({img_md} "title-2")

| Train        | Val           | Test  |
| ------------- |:-------------:| -----:|
| {img_md}      | {img_md}      | {img_md} |



---

### Summary Statistics

- Mean bbox area norm: **{df['bbox_area_norm'].mean():.4f}**
- Median bbox area norm: **{df['bbox_area_norm'].median():.4f}**
- Std bbox area norm: **{df['bbox_area_norm'].std():.4f}**
"""

    # Save Markdown
    with open(md_path, "w") as f:
        f.write(md_content)

    # ---------------------------------------------------------
    # 4. Convert Markdown → HTML
    # ---------------------------------------------------------
    html = markdown(md_content, output_format="html5")

    # ---------------------------------------------------------
    # 5. Convert HTML → PDF
    # ---------------------------------------------------------
    HTML(string=html).write_pdf(pdf_path)

    print("Markdown saved:", md_path)
    print("PDF saved:", pdf_path)
    return md_content

#%%

df = pd.DataFrame({
    "bbox_area_norm": [0.1, 0.2, 0.05, 0.3, 0.15, 0.22, 0.18]
})

datacard_md = generate_dataset_card_with_embedded_plot(df)


fig = px.histogram(
        df,
        x="bbox_area_norm",
        title="BBox Area Distribution",
        nbins=20
    )

img_bytes = fig.to_image(format="png")  # requires kaleido
img_b64 = base64.b64encode(img_bytes).decode("utf-8")



#%%


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import base64

def make_split_plot(splits: dict):
    """
    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }
    """
    num = len(splits)
    fig = make_subplots(rows=1, cols=num, subplot_titles=[k.capitalize() for k in splits])

    col = 1
    for name, df in splits.items():
        fig_sub = px.histogram(df, x="bbox_area_norm", nbins=20)
        for trace in fig_sub.data:
            fig.add_trace(trace, row=1, col=col)
        col += 1

    fig.update_layout(height=400, width=300*num, showlegend=False)
    return fig


def fig_to_base64(fig):
    img_bytes = fig.to_image(format="png")
    return base64.b64encode(img_bytes).decode("utf-8")


def generate_dataset_card(df_splits, md_path="dataset_card.md", pdf_path="dataset_card.pdf"):
    fig = make_split_plot(df_splits)
    img_b64 = fig_to_base64(fig)
    img_md = f"![splits](data:image/png;base64,{img_b64})"

    md_content = f"""
# Dataset Card

## BBox Area Distribution by Split

{img_md}

---
"""

    with open(md_path, "w") as f:
        f.write(md_content)

    # Convert Markdown → HTML → PDF (your existing pipeline)
    from markdown import markdown
    from weasyprint import HTML

    html = markdown(md_content, output_format="html5")
    HTML(string=html).write_pdf(pdf_path)
    
    return md_content


df_train = pd.DataFrame({"bbox_area_norm": [0.1, 0.2, 0.15, 0.3]})
df_val   = pd.DataFrame({"bbox_area_norm": [0.05, 0.12, 0.18]})
df_test  = pd.DataFrame({"bbox_area_norm": [0.2, 0.25, 0.22, 0.3]})

splits = {"train": df_train, "val": df_val, "test": df_test}

card_md2 = generate_dataset_card(splits)


from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.write_html(card_md2)
pdf.output("dataset_card_subplots.pdf")

#%%

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.image(BytesIO(base64.b64decode(img_b64)), x=10, y=10, w=100)  # Example of adding the image to PDF
pdf.write_html(datacard_md)
pdf.output("dataset_card2.pdf")


#%%

from pprint import pprint
from markdown_it import MarkdownIt
md = MarkdownIt()
md.render("some *text*")

md
#%%



import plotly.express as px
import pandas as pd
from markdown import markdown
from weasyprint import HTML
import base64


def fig_to_b64(fig):
    """Convert Plotly figure to base64 PNG."""
    img_bytes = fig.to_image(format="png")
    return base64.b64encode(img_bytes).decode("utf-8")


def generate_dataset_card_with_multiple_plots(
    dfs: dict,   # {"train": df_train, "val": df_val, ...}
    md_path="dataset_card.md",
    pdf_path="dataset_card.pdf"
):
    # ---------------------------------------------------------
    # 1. Create base64 images for each split
    # ---------------------------------------------------------
    plot_blocks = []
    num_plots = len(dfs)
    width = 100 / num_plots  # divide page width evenly

    for name, df in dfs.items():
        fig = px.histogram(
            df,
            x="bbox_area_norm",
            title=f"{name.capitalize()} BBox Area Distribution",
            nbins=20
        )

        img_b64 = fig_to_b64(fig)

        block = f"""
        <div style="display:inline-block; width:{width}%; text-align:center; vertical-align:top;">
            <h4>{name.capitalize()}</h4>
            <img src="data:image/png;base64,{img_b64}" style="width:95%;"/>
        </div>
        """

        plot_blocks.append(block)

    plots_html = "\n".join(plot_blocks)

    # ---------------------------------------------------------
    # 2. Build Markdown content
    # ---------------------------------------------------------
    md_content = f"""
# Dataset Card

## Data-Centric Metrics

### BBox Area Distribution by Split

{plots_html}

---

### Summary Statistics
"""

    for name, df in dfs.items():
        md_content += f"""
#### {name.capitalize()}
- Mean: **{df['bbox_area_norm'].mean():.4f}**
- Median: **{df['bbox_area_norm'].median():.4f}**
- Std: **{df['bbox_area_norm'].std():.4f}**
"""

    # Save Markdown
    with open(md_path, "w") as f:
        f.write(md_content)

    # ---------------------------------------------------------
    # 3. Convert Markdown → HTML
    # ---------------------------------------------------------
    html = markdown(md_content, output_format="html5")

    # ---------------------------------------------------------
    # 4. Convert HTML → PDF
    # ---------------------------------------------------------
    HTML(string=html).write_pdf(pdf_path)

    print("Markdown saved:", md_path)
    print("PDF saved:", pdf_path)



#%%
df_train = pd.DataFrame({"bbox_area_norm": [0.1, 0.2, 0.15, 0.3]})
df_val   = pd.DataFrame({"bbox_area_norm": [0.05, 0.12, 0.18]})
df_test  = pd.DataFrame({"bbox_area_norm": [0.2, 0.25, 0.22, 0.3]})

splits = {
    "train": df_train,
    "val": df_val,
    "test": df_test
}

generate_dataset_card_with_multiple_plots(splits)

#%%

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', size=12)
pdf.cell(text="hello world")
pdf.output("hello_world.pdf")

#%%

from mistletoe import markdown

html = markdown("""
# Top title (ATX)

Subtitle (setext)
-----------------

### An even lower heading (ATX)

**Text in bold**

_Text in italics_

~~Strikethrough~~

[This is a link](https://github.com/PyFPDF/fpdf2)

<https://py-pdf.github.io/fpdf2/>

This is an unordered list:
* an item
* another item

This is an ordered list:
1. first item
2. second item
3. third item with an unordered sublist:
    * an item
    * another item

Inline `code span`

A table:

| Foo | Bar | Baz |
| ---:|:---:|:--- |
| Foo | Bar | Baz |

Actual HTML:

<dl>
  <dt>Term1</dt><dd>Definition1</dd>
  <dt>Term2</dt><dd>Definition2</dd>
</dl>

Some horizontal thematic breaks:

***
---
___

![Alternate description](https://py-pdf.github.io/fpdf2/fpdf2-logo.png)
""")

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.write_html(html)
pdf.output("pdf-from-markdown-with-mistletoe.pdf")









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
    
    