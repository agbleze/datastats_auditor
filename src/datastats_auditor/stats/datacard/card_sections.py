
import base64


def fig_to_base64(fig):
    img_bytes = fig.to_image(format="png")
    return base64.b64encode(img_bytes).decode("utf-8")
    

def create_section(*args, **kwargs):
    kwargs = kwargs.get("kwargs")
    header = kwargs.get("header", "")
    header = f"{header}" if header else ""
    sect_content = "\n".join([f"- {key}: {value}" for key, value in kwargs.items() if key != "header"])
    section = f""" 
<H1 style="color:#2A7FFF; font-weight:bold;">
 {header}
</H1>

{sect_content}
    
    """
    return section.strip()


def create_section_head(**kwargs):
    header = kwargs.get("header", "")
    header = f"{header}" if header else ""
    section_content = kwargs.get("section", "")
    section = f""" 
<H1 style="color:#2A7FFF; font-weight:bold;">
 {header}
</H1>

{section_content}
    
    """
    return section.strip()
    

    
def generate_data_metric_section(metric_heading, fig,
                                 subheading="", footnote="",
                                 ):
    img_b64 = fig_to_base64(fig)
    img_md = f"![splits](data:image/png;base64,{img_b64})"

    md_content = f"""## {metric_heading}
    
### {subheading}

{img_md}
 
{footnote}
    """
    return md_content
    


def create_drift_section(drift_result, metric_used="Jensen Shannon Divergence",
                            drift_key='js_2d', 
                            
                            ):
    section_list = []
    drift_plot = drift_result["drift_plot"]
    drift_spatial_heatmap = drift_result["spatial_heatmap"]
    spatial_drift = drift_result['spatial_drift']
    for idx, (distr_pair, fig) in enumerate(drift_plot.items()):
        heading = "Data Drift Detection on Data Attributes" if idx == 0 else ""
        
        sec = generate_data_metric_section(metric_heading=heading,
                                    fig=fig,
                                    subheading=f"{metric_used} on {distr_pair} Distribution",
                                    footnote="Score > 0.05 indicates statistically significant drift",
                                    )
        section_list.append(sec)
    
    for idx, (distr_pair, fig) in enumerate(drift_spatial_heatmap.items()):
        pair_spatial_dirft = spatial_drift[distr_pair]
        spatial_drift_content = generate_data_metric_section(metric_heading="", 
                                                            fig=fig, 
                                                            subheading=f"Spatial Drift: Relative Object Centers - {distr_pair}", 
                                                            footnote=f"{metric_used}: {pair_spatial_dirft[drift_key]: .3f} | Wasserstein: {pair_spatial_dirft['w1_2d']: .3f}"  
                                                            )
        section_list.append(spatial_drift_content)
    
    drift_section_content = "\n".join([i for i in section_list])
    return drift_section_content



                
def create_scene_composition_section(scene_results, 
                                        data_properites=["occupancy_per_image",
                                                        "relative_bbox_area",
                                                        "bbox_aspect_ratio"
                                                        ],
                                        ):
    contents_list = []
    for idx, prop in enumerate(data_properites):
        heading = "Scene composition" if idx == 0 else ""
        fig = scene_results[prop]
        scene_content = generate_data_metric_section(metric_heading=heading,
                                                    fig=fig,
                                                    subheading=f"Distribution of {prop}", 
                                                    footnote=f""  
                                                    )
        contents_list.append(scene_content)
    scene_contents = "\n".join([i for i in contents_list])
    return scene_contents
        