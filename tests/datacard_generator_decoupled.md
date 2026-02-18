
## 1. Core structure

```text
dataset_card/
│
├── __init__.py
├── dataset_card_creator.py
├── cli.py
│
├── sections/
│   ├── __init__.py
│   ├── summary.py
│   ├── motivation.py
│   ├── composition.py
│   ├── collection.py
│   ├── annotation.py
│   ├── data_centric_metrics.py
│   ├── splits.py
│   ├── ethics.py
│   ├── licensing.py
│   └── maintenance.py
│
└── renderer/
    ├── __init__.py
    ├── markdown_renderer.py
    └── html_renderer.py
```

---

## 2. Core creator + renderers

### `dataset_card_creator.py`

```python
# dataset_card/dataset_card_creator.py
from pathlib import Path

class DatasetCardCreator:
    """
    Orchestrates sections + renderer to produce a dataset card.
    """

    def __init__(self, sections: dict, renderer):
        """
        sections: dict[str, str] mapping section titles to content
        renderer: object with .render(sections: dict) -> str
        """
        self.sections = sections
        self.renderer = renderer

    def generate(self, output_path: str = "DATASET_CARD.md"):
        card_text = self.renderer.render(self.sections)
        Path(output_path).write_text(card_text, encoding="utf-8")
        print(f"Dataset card created at: {output_path}")
```

### `renderer/markdown_renderer.py`

```python
# dataset_card/renderer/markdown_renderer.py

class MarkdownRenderer:
    """
    Renders sections into a clean Markdown dataset card.
    """

    def render(self, sections: dict) -> str:
        md = "# 📘 Dataset Card\n\n"
        for title, content in sections.items():
            md += f"## {title}\n\n"
            md += content.strip() + "\n\n"
        return md
```

### `renderer/html_renderer.py`

```python
# dataset_card/renderer/html_renderer.py

class HTMLRenderer:
    """
    Simple HTML renderer. You can later feed this into a PDF generator (WeasyPrint, wkhtmltopdf, etc.).
    """

    def render(self, sections: dict) -> str:
        html = [
            "<html>",
            "<head><meta charset='utf-8'><title>Dataset Card</title></head>",
            "<body style='font-family: sans-serif; max-width: 900px; margin: auto;'>",
            "<h1>📘 Dataset Card</h1>",
        ]
        for title, content in sections.items():
            html.append(f"<h2>{title}</h2>")
            # naive conversion: line breaks → <p> blocks
            for block in content.strip().split("\n\n"):
                html.append(f"<p>{block}</p>")
        html.append("</body></html>")
        return "\n".join(html)
```

---

## 3. Section templates

Each section module returns a string. You can later make them classes if you want.

### `sections/summary.py`

```python
# dataset_card/sections/summary.py
from datetime import datetime

def build_summary(dataset_name: str, num_images: int, num_annotations: int, num_classes: int | str):
    return f"""
**Dataset name:** {dataset_name}  
**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This dataset is an object detection dataset with {num_images} images, {num_annotations} annotations, and {num_classes} classes.
"""
```

### `sections/motivation.py`

```python
# dataset_card/sections/motivation.py

def build_motivation(intended_uses: str, out_of_scope_uses: str, motivation_text: str):
    return f"""
**Motivation**

{motivation_text}

- **Intended uses:** {intended_uses}  
- **Out-of-scope uses:** {out_of_scope_uses}
"""
```

### `sections/composition.py`

```python
# dataset_card/sections/composition.py

def build_composition(class_stats: str, modalities: str = "Images + bounding boxes"):
    return f"""
**Modalities:** {modalities}

**Class distribution:**

{class_stats}
"""
```

### `sections/collection.py`

```python
# dataset_card/sections/collection.py

def build_collection(collection_process: str):
    return collection_process
```

### `sections/annotation.py`

```python
# dataset_card/sections/annotation.py

def build_annotation(annotation_process: str):
    return annotation_process
```

### `sections/data_centric_metrics.py` (auto-filling from your df)

```python
# dataset_card/sections/data_centric_metrics.py

import pandas as pd

def build_data_centric_metrics(df: pd.DataFrame) -> str:
    per_image = df.drop_duplicates("image_id")

    def fmt(x, d=4):
        try:
            return f"{x:.{d}f}"
        except:
            return "N/A"

    return f"""
### Per-object metrics

- **`bbox_area_norm`** — normalized bbox area relative to image area  
- **`bbox_aspect_ratio`** — width / height  
- **`center_x_norm`, `center_y_norm`** — normalized bbox center coordinates  

**Summary:**

- Mean bbox area norm: {fmt(df["bbox_area_norm"].mean())}  
- Median bbox area norm: {fmt(df["bbox_area_norm"].median())}  
- Mean aspect ratio: {fmt(df["bbox_aspect_ratio"].mean())}  

---

### Per-image metrics

- **`num_bboxes_per_image`** — number of objects per image  
- **`bbox_area_norm_variance_per_image`** — variance of object sizes within each image  
- **`occupancy_per_image`** — sum of bbox areas / image area  
- **`foreground_union_area_per_image`** — union-based foreground area  
- **`background_area_norm`** — background area / image area  
- **`foreground_to_background_area_per_image`** — union-based foreground/background contrast  
- **`foreground_occupancy_to_background_occupancy`** — sum-based foreground/background contrast  

**Summary:**

- Mean objects per image: {fmt(per_image["num_bboxes_per_image"].mean(), 2)}  
- Mean bbox size variance: {fmt(per_image["bbox_area_norm_variance_per_image"].mean())}  
- Mean foreground ratio (union): {fmt(per_image["foreground_union_area_per_image"].mean())}  
- Mean occupancy (sum of areas): {fmt(per_image["occupancy_per_image"].mean())}  
- Mean fg/bg contrast (union): {fmt(per_image["foreground_to_background_area_per_image"].mean())}  

---

### Interpretation

- **Scale drift:** via bbox area stats and variance  
- **Composition drift:** via number of objects per image  
- **Scene density drift:** via occupancy metrics  
- **Foreground-background structure:** via union vs sum contrast  
- **Annotation style drift:** via differences between union-based and sum-based metrics  
"""
```

### `sections/splits.py`

```python
# dataset_card/sections/splits.py

def build_splits(split_stats: str):
    return split_stats
```

### `sections/ethics.py`

```python
# dataset_card/sections/ethics.py

def build_ethics(ethical_considerations: str):
    return ethical_considerations
```

### `sections/licensing.py`

```python
# dataset_card/sections/licensing.py

def build_licensing(license_text: str):
    return license_text
```

### `sections/maintenance.py`

```python
# dataset_card/sections/maintenance.py

def build_maintenance(contact: str, versioning: str, known_issues: str):
    return f"""
**Contact:** {contact}  

**Versioning:**  
{versioning}

**Known issues:**  
{known_issues}
"""
```

---

## 4. CLI entrypoint

### `cli.py`

```python
# dataset_card/cli.py

import argparse
import pandas as pd

from dataset_card_creator import DatasetCardCreator
from renderer.markdown_renderer import MarkdownRenderer
from sections.summary import build_summary
from sections.motivation import build_motivation
from sections.composition import build_composition
from sections.collection import build_collection
from sections.annotation import build_annotation
from sections.data_centric_metrics import build_data_centric_metrics
from sections.splits import build_splits
from sections.ethics import build_ethics
from sections.licensing import build_licensing
from sections.maintenance import build_maintenance


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset card.")
    parser.add_argument("--df-path", type=str, required=True, help="Path to a pickled or parquet DataFrame with metrics.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output", type=str, default="DATASET_CARD.md")
    args = parser.parse_args()

    # Load df (adapt as needed)
    if args.df_path.endswith(".pkl"):
        df = pd.read_pickle(args.df_path)
    elif args.df_path.endswith(".parquet"):
        df = pd.read_parquet(args.df_path)
    else:
        raise ValueError("Unsupported df format. Use .pkl or .parquet")

    num_images = df["image_id"].nunique()
    num_annotations = len(df)
    num_classes = df["category_id"].nunique() if "category_id" in df.columns else "N/A"

    # Here you can later load these from config / YAML / UI
    summary = build_summary(args.dataset_name, num_images, num_annotations, num_classes)
    motivation = build_motivation(
        intended_uses="Object detection research and benchmarking.",
        out_of_scope_uses="Sensitive decision-making without human oversight.",
        motivation_text="This dataset was created to support robust evaluation of object detection models under varying conditions."
    )
    composition = build_composition(class_stats="(Add class distribution table or text here.)")
    collection = build_collection("Describe how the data was collected here.")
    annotation = build_annotation("Describe how annotations were created and validated here.")
    metrics = build_data_centric_metrics(df)
    splits = build_splits("Describe train/val/test splits and any drift analysis here.")
    ethics = build_ethics("Describe potential biases, risks, and limitations here.")
    licensing = build_licensing("Specify license, allowed uses, and restrictions here.")
    maintenance = build_maintenance(
        contact="your-email@example.com",
        versioning="Version 1.0. Future updates will be documented here.",
        known_issues="List known issues or limitations here."
    )

    sections = {
        "1. Dataset Summary": summary,
        "2. Motivation": motivation,
        "3. Composition": composition,
        "4. Collection Process": collection,
        "5. Annotation Process": annotation,
        "6. Data-Centric ML Quality Metrics": metrics,
        "7. Dataset Splits": splits,
        "8. Ethical Considerations": ethics,
        "9. Licensing & Usage": licensing,
        "10. Maintenance": maintenance,
    }

    renderer = MarkdownRenderer()
    creator = DatasetCardCreator(sections, renderer)
    creator.generate(args.output)


if __name__ == "__main__":
    main()
```

Run like:

```bash
python -m dataset_card.cli \
  --df-path metrics.pkl \
  --dataset-name "My Detection Dataset" \
  --output DATASET_CARD.md
```

---

## 5. Plots & JS drift hooks (where to plug them)

You don’t need full code now—just clear anchor points.

- Add a `plots/` module that generates:
  - histograms for bbox_area_norm, num_bboxes_per_image, etc.  
  - JS drift tables between splits  
- Save plots as PNGs and:
  - link them in Markdown (`![title](path/to/plot.png)`)  
  - or embed `<img>` tags in HTML renderer  

You can extend `build_data_centric_metrics` to append:

```markdown
![BBox area distribution](plots/bbox_area_norm_hist.png)

![Num bboxes per image](plots/num_bboxes_per_image_hist.png)
```
