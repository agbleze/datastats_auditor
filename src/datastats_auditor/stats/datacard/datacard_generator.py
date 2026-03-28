from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
import plotly.express as px
from .card_sections import (create_section, 
                            create_drift_section, 
                            create_scene_composition_section,
                            generate_data_metric_section,
                            create_drift_section,
                            create_section_head
                            )
from .utils import (compute_summary_stats_wider, 
                    concat_split_dfs,
                    create_data_overview
                    )
from .visualizer import (plot_summary_table,
                        plot_bar, HistPlot
                        )
from .constants import (AUTHORSHIP,
                        DATA_COLLECTION,
                        LABELLING, LICENCE,
                        TRANSFORMATION, DATA_SPLIT
                        )


class DatasetCardCreator:
    def __init__(self, sections: dict, renderer):
        self.sections = sections
        self.renderer = renderer

    def generate(self, output_path: str = "DATASET_CARD.md",
                 export_pdf: bool = True
                 ):
        self.text = self.renderer.render(self.sections)
        Path(output_path).write_text(self.text, encoding="utf-8")
        print(f"Dataset card created at: {output_path}")
        
        if export_pdf:
            pdf_path = f"{output_path.split('.md')[0]}.pdf"
            export_md_to_pdf(md=self.text, save_path=pdf_path)
        return self.text
        
def build_motivations_and_use(purposes=None,
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

def export_md_to_pdf(md, save_path="datacard_2.pdf"):
    pdf = MarkdownPdf()

    pdf.add_section(Section(md))
    pdf.save(save_path)


class MarkdownRenderer:
    def render(self, sections: dict) -> str:
        md = "# 📘 Dataset Card\n\n"
        for title, content in sections.items():
            if content is None or not str(content).strip():
                continue
            md += f"## {title}\n\n{content.strip()}\n\n"
        return md

def create_data_card(split_stats_result,
                     drift_result,
                     version_id, card_name,
                     intended_objects=None,
                     **kwargs
                    ):
    summary_stats_df = compute_summary_stats_wider(split_stats_result)
    wider_summary_table_fig = plot_summary_table(summary_stats_df, height=300)
    full_split_df = concat_split_dfs(split_dfs=split_stats_result.split_dfs)

    object_dist_barplot = plot_bar(full_split_df.groupby(["split_type", "category_name"]).size().reset_index(name="count"),
                                    x="category_name", y="count",
                                    title="Category Distribution by Split",
                                    facet_row="split_type",
                                    color="split_type",
                                    color_discrete_sequence=px.colors.qualitative.Plotly,
                                    labels={"category_name": "Category", "count": "Count", "split_type": "Split"},
                                    text="count",
                                    height=500, width=700,
                                    )
    drift_section = create_drift_section(drift_result=drift_result)

    histplot_cls = HistPlot(df=full_split_df, 
                            property_names=["relative_bbox_area", 'bbox_aspect_ratio',
                                            'occupancy_per_image', 'num_bboxes_per_image',
                                            ],
                            facet_row="split_type",
                            height=500, width=700,
                            )
    hist_figs = histplot_cls.create_histograms()
    scene_composition_section = create_scene_composition_section(hist_figs)
    drift_scene_section = "\n".join([i for i in [drift_section, scene_composition_section]])
    intended_objects = list(full_split_df["category_name"].unique()) if intended_objects is None else intended_objects 
    moti_content = build_motivations_and_use(intended_tasks=kwargs.get("intended_tasks"),
                                            intended_objects=intended_objects, #["Person", "Helmet", "Safety vest"],
                                            )
    object_bias_content = generate_data_metric_section(metric_heading=f"Object Balance per Split",
                                                        fig=object_dist_barplot,
                                                        )
    summary_kwargs = create_data_overview(split_dfs=split_stats_result.split_dfs, 
                                            name=card_name, id=version_id
                                            )
    summary_section = create_section(kwargs=summary_kwargs)
    authorship_section = create_section(kwargs=AUTHORSHIP)
    data_collection_section = create_section(kwargs=DATA_COLLECTION)
    labelling_section = create_section(kwargs=LABELLING)
    split_section = create_section(kwargs=kwargs.get("split_data", DATA_SPLIT))
    transformation_section = create_section(kwargs=kwargs.get("transformation", TRANSFORMATION))
    license_section = create_section(kwargs=LICENCE)
    wider_summary_table_fig_base64 = generate_data_metric_section(metric_heading="",
                                                                    fig=wider_summary_table_fig
                                                                    )
    title = create_section_head(header="Data Card")
    data_overview = create_section_head(header="Data Overview",
                                        section=summary_section
                                        )
    wider_summary_statistics_section = create_section_head(header="Descriptive Summary Statistics",
                                                            section=wider_summary_table_fig_base64
                                                            )
    dataspace_metrics = create_section_head(header="Data Space Metrics",
                                            section=drift_scene_section, 
                                            )
    motivation_section = create_section_head(header="Motivation and Intended Use",
                                            section=moti_content
                                            )
    joined_section = "\n".join([title, 
                                authorship_section, 
                                motivation_section, 
                                data_overview,
                                data_collection_section, 
                                labelling_section, 
                                split_section,
                                wider_summary_statistics_section, 
                                object_bias_content,
                                dataspace_metrics, 
                                transformation_section,
                                license_section
                                ])
    return joined_section