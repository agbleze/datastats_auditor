from markdown import markdown
from weasyprint import HTML
import pandas as pd
from plotly.subplots import make_subplots
import itertools
from datastats_auditor import logger

    
def write_md_to_pdf(markdown_content: str, pdf_path: str, **kwargs):
    html = markdown(markdown_content, output_format="html5")
    HTML(string=html).write_pdf(pdf_path)
        

def concat_split_dfs(split_dfs: dict):
    df_list = []
    for split_nm, df in split_dfs.items():
        df["split_type"] = split_nm
        df_list.append(df)
        
    split_df = pd.concat(df_list)
    return split_df

def compute_stats(df, prop, group="category_name",
                  stats=["mean", "std", "min", "max", "median"]
                  ):
    if group:
        df = df.groupby(group)
    return (df[prop]
          .agg(stats).reset_index()
            )

def compute_summary_stats_wider(result, properties=["relative_bbox_area",
                                                "bbox_aspect_ratio",
                                                "num_bboxes_per_image"
                                                ],
                              **kwargs
                            ):
    stats = ["mean", "min", "max", "median", "std", "skew", "kurt"]
    split_dfs = result.split_dfs
    logger.info(f"split_dfs: {split_dfs}")
    stats_df_list = []
    for split_nm, split_df in split_dfs.items():
        for col in properties:
            df = compute_stats(df=split_df, prop=col, 
                            group=kwargs.get("group"),
                            stats=kwargs.get("stats", stats)
                                )
            df = df.set_index("index").T
            df = df.round(kwargs.get("round", 5))
            df["attribute"] = f"{split_nm}_{df.index[0]}"
            logger.info(f"{split_nm}: {df}")
            stats_df_list.append(df)
    summary_stat_df = pd.concat(stats_df_list)
    return summary_stat_df

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


def create_data_overview(split_dfs: dict, name, id):
    num_images = {split_nm: df.image_id.nunique() for split_nm, df in 
                    split_dfs.items()
                    }
    split_obj_num = {split_nm: df.category_id.count() for split_nm, df in 
                    split_dfs.items()
                    }
    all_categories = []
    
    for split_nm, df in split_dfs.items():
        all_categories.extend(list(df.category_name.unique()))
    all_categories = set(all_categories)
    
    summary_kwargs = {"Name": name,
                        "Version ID": id,
                        "Modality": "Image",
                        "Number of images": num_images,
                        "Objects labeled": all_categories, 
                        "Object counts per split": split_obj_num 
                        }
    return summary_kwargs
