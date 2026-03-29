import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

def plot_split_bin_bars(per_split_ratios, title="Split Size-Bin Ratios"):
    """
    per_split_ratios: DataFrame index=split, columns=bins, values=ratio
    """
    df = per_split_ratios.reset_index().melt(
        id_vars="split", var_name="bin", value_name="ratio"
    )

    fig = px.bar(
        df,
        x="bin",
        y="ratio",
        color="split",
        barmode="group",
        title=title,
        text_auto=".2f"
    )
    fig.update_layout(xaxis_title="Size Bin", yaxis_title="Ratio")
    return fig

def plot_class_bin_bars(per_class_ratios, title="Per-Class Size-Bin Ratios"):
    """
    per_class_ratios: DataFrame index=class, columns=bins, values=ratio
    """
    df = per_class_ratios.reset_index().melt(id_vars="category_name", 
                                             var_name="bin", 
                                             value_name="ratio"
                                             )

    fig = px.bar(df,
                x="category_name",
                y="ratio",
                color="bin",
                barmode="stack",
                title=title,
                text_auto=".2f"
                )
    fig.update_layout(xaxis_title="Class", yaxis_title="Ratio")
    fig.update_xaxes(tickangle=45)
    return fig

def plot_area_kde(df, title="KDE of Normalized BBox Area"):
    areas = df["bbox_area_norm"].clip(1e-9, 1.0)
    kde = gaussian_kde(areas)
    xs = np.logspace(-9, 0, 400)
    ys = kde(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy"))
    fig.update_layout(title=title,
                        xaxis=dict(title="bbox_area_norm (log scale)", type="log"),
                        yaxis=dict(title="Density")
                    )
    return fig

def plot_area_hist_log(df, bins=60, title="Log-Log Histogram of BBox Area"):
    areas = df["bbox_area_norm"].clip(1e-9, 1.0)

    fig = px.histogram(x=areas,
                        nbins=bins,
                        log_x=True,
                        log_y=True,
                        title=title
                        )
    fig.update_layout(xaxis_title="bbox_area_norm (log)",
                        yaxis_title="Count (log)"
                    )
    return fig

from sklearn.cluster import KMeans

def suggest_anchors(df, n_anchors=9, normalize=True, random_state=0):
    # width/height in pixels
    w = df["bbox_w"].values
    h = df["bbox_h"].values

    if normalize:
        w = w / df["image_width"].values
        h = h / df["image_height"].values

    X = np.stack([w, h], axis=1)

    kmeans = KMeans(n_clusters=n_anchors, n_init=10, random_state=random_state)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # sort by area
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    anchors = centers[order]

    return anchors  # shape (n_anchors, 2) -> (w, h) normalized or absolute

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def split_kl_drift(per_split_ratios):
    """
    per_split_ratios: DataFrame index=split, columns=bins, values=ratio
    returns dict of pairwise KL divergences between splits
    """
    splits = per_split_ratios.index.tolist()
    kl_results = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            p = per_split_ratios.loc[s1].values
            q = per_split_ratios.loc[s2].values
            kl_results[(s1, s2)] = kl_divergence(p, q)
    return kl_results

def per_class_split_kl(per_class_split_ratios):
    """
    per_class_split_ratios: MultiIndex (category_name, split) × bins
    returns DataFrame: index=category_name, columns=(split1, split2) pairs
    """
    classes = per_class_split_ratios.index.get_level_values(0).unique()
    splits = per_class_split_ratios.index.get_level_values(1).unique()

    results = {}
    for cls in classes:
        cls_ratios = per_class_split_ratios.loc[cls]  # split × bins
        cls_kl = {}
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                p = cls_ratios.loc[s1].values
                q = cls_ratios.loc[s2].values
                cls_kl[(s1, s2)] = kl_divergence(p, q)
        results[cls] = cls_kl

    # convert to DataFrame with MultiIndex columns (split1, split2)
    all_pairs = sorted({pair for cls in results for pair in results[cls].keys()})
    out = pd.DataFrame(index=classes, columns=pd.MultiIndex.from_tuples(all_pairs))
    for cls, d in results.items():
        for pair, val in d.items():
            out.loc[cls, pair] = val
    return out

def plot_spatial_heatmaps(spatial_dict_A, spatial_dict_B, 
                          names=("A", "B"),
                          **kwargs
                          ):
    H_A = spatial_dict_A["heatmap"]
    H_B = spatial_dict_B["heatmap"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"{names[0]} Heatmap", f"{names[1]} Heatmap"]
                        )

    fig.add_trace(go.Heatmap(z=H_A, colorscale="Viridis"),
                    row=1, col=1
                )
    fig.add_trace(go.Heatmap(z=H_B, colorscale="Viridis"),
                    row=1, col=2
                    )
    annotations = []
    for i in range(H_A.shape[0]): 
        for j in range(H_A.shape[1]): 
            annotations.append( dict( x=j, y=i, xref="x1", yref="y1", 
                                     text=f"{H_A[i, j]:.3f}", 
                                     showarrow=False, 
                                     font=dict( color="black", 
                                               size=7
                                               ) 
                                     ) 
                               )
    for i in range(H_B.shape[0]): 
        for j in range(H_B.shape[1]): 
            annotations.append( dict( x=j, y=i, xref="x2", yref="y2", text=f"{H_B[i, j]:.3f}", 
                                     showarrow=False, 
                                     font=dict( color="black",
                                               size=7 
                                               ) 
                                     ) 
                               )
    fig.update_layout(height=kwargs.get("height", 400),
                        width=kwargs.get("width", 700),
                        coloraxis=dict(colorscale="Viridis"),
                        showlegend=False,
                        annotations=annotations,
                        template=kwargs.get("template", "plotly_dark"),
                        title=kwargs.get("title_text")
                        )
    fig.update_yaxes(autorange="reversed", row=1, col=1) 
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    return fig

def plot_drift_radar(drift_df, drift_scores_colname: List[float], 
                     drift_properties_colname: List[str],
                     **kwargs
                     ):
    fig = px.line_polar(drift_df,r=drift_scores_colname, 
                        theta=drift_properties_colname,
                        template=kwargs.get("template","plotly_dark"),
                        title=kwargs.get("title"),
                        width=kwargs.get("width", 700),
                        height=kwargs.get("height", 500)
                        )
    return fig
  
def get_drift_result_as_df(drift_results, 
                            distribution_pair: str = "('train', 'val')",
                            property_field_name: str = "property",
                            metric_name: str = "js"
                            ):
    drift_results.keys()
    drift_property = []
    drift_metric = []
    for k, v in drift_results.items():
        if k.startswith(distribution_pair) and k.endswith(metric_name):
            drift_property.append(v[property_field_name])
            drift_metric.append(v[metric_name])
            
    drift_dict = {"scores": drift_metric, 
                "property": drift_property
                }
    df = pd.DataFrame(drift_dict)
    return df