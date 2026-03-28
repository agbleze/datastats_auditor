import plotly.express as px
from typing import Union
import plotly.graph_objects as go
import pandas as pd


def plot_histogram(df, **kwargs):
    fig = px.histogram(df, x=kwargs.get("x"), 
                        histnorm=kwargs.get("histnorm"), 
                        title=kwargs.get("title"), 
                        template=kwargs.get("template", "plotly_dark"),
                        color=kwargs.get("color"),
                        facet_col=kwargs.get("facet_col"),
                        facet_row=kwargs.get("facet_row"),
                        facet_col_spacing=kwargs.get("facet_col_spacing", 0.1),
                        height=kwargs.get("height", 800),
                        width=kwargs.get("width",800),
                        barmode=kwargs.get("barmode", "relative"),
                        
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
    return fig

            
class HistPlot:
    def __init__(self, df, property_names: Union[str, list], **kwargs):
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names
        self.kwargs = kwargs
        self.df = df
        
    def create_histograms(self):
        self.property_histograms = {}
        for prop in self.property_names:
            prop_fig = plot_histogram(df=self.df, x=prop,
                                        **self.kwargs
                                        )
            self.property_histograms[prop] = prop_fig
        return self.property_histograms

def compute_column_widths(df, min_width=80, max_width=300):
    widths = []
    for col in df.columns:
        values = df[col].astype(str)
        max_len = max([len(col)] + [len(v) for v in values])
        width = max(min_width, min(max_width, max_len * 9))
        widths.append(width)
    return widths

def plot_summary_table(df, title="Summary Statistics", **kwargs):
    columns = list(df.columns[::-1])
    col_widths = compute_column_widths(df=df)[::-1]
    fig = go.Figure(data=[go.Table(columnwidth=col_widths,
                            header=dict(values=columns,
                                        fill_color="#1f2c56",
                                        font=dict(color="white", size=12),
                                        align="left"
                                        ),
                            cells=dict(values=[df[col] for col in columns], 
                                        fill_color="#2d3e6b",
                                        font=dict(color="white", size=11),
                                        align="left",
                                    )
                            )
                        ]
                    )

    fig.update_layout(title=kwargs.get("title", title),
                        height=kwargs.get("height"),
                        width=kwargs.get("width", 700),
                        margin=kwargs.get("margin",dict(l=10, r=10, t=40, b=10)),
                        template=kwargs.get("template","plotly_dark")
                    )

    return fig