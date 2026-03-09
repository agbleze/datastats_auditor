
import pandas as pd
import uuid
import plotly.express as px


train_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/train/_annotations.coco.json"
val_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/valid/_annotations.coco.json"
test_annfile = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/test/_annotations.coco.json"


train_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/train"
val_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/valid"
test_imgdir = "/home/lin/codebase/tomato_disease_prediction/Tomato-pest&diseases-1/test"

obj_stats_kwargs = {"train": {"coco_ann": train_annfile, "n_bins": 5, "strategy": "quantile"},
                    "val": {"coco_ann": val_annfile, "n_bins": 5, "strategy": "quantile"},
                    "test": {"coco_ann": test_annfile, "n_bins": 5, "strategy": "quantile"}
                }

image_stats_kwargs = {"train": {"image_dir": train_imgdir},
                        "val": {"image_dir": val_imgdir},
                        "test": {"image_dir": test_imgdir}
                    }

intended_tasks=["Object detection", "Fairness evaluation"]
PDF_PATH = "dataset_card.pdf"
#%%

print(f"obj_stats_kwargs: {obj_stats_kwargs}")

#%%
split_stats_cls = SplitStats(object_stats_cls=ObjectStats,
                            image_loader_cls=ImageBatchDataset, 
                            object_stats_kwargs=obj_stats_kwargs,
                            image_stats_kwargs=image_stats_kwargs
                            )  



#%%

split_stats_res = split_stats_cls.compute_stats()


summary_stats_df = compute_summary_stats_wider(split_stats_res)
    
    
#%%


#%%


#%% Example usage:
wider_summary_table_fig = plot_summary_table(summary_stats_df, height=300)
wider_summary_table_fig

    
distributions = split_stats_res["split_dfs"]

metrics = ["kl", "js"]

field_to_bin = ['relative_bbox_area', 'bbox_aspect_ratio',
                'relative_x_center', 'relative_y_center',
                'foreground_union_area_per_image',
                'occupancy_per_image', 'background_area_per_image',
                #'foreground_to_background_area_per_image', 
                'background_area_norm',
                #'foreground_occupancy_to_background_occupany', 
                'num_bboxes_per_image',
                'relative_bbox_area_variance_per_image'
                ]

#%%

drift_suite_cls = DriftMetricSuite(distributions=distributions,
                                    metrics=metrics, field_to_bin=field_to_bin,
                                    #name_bin_field_as=None,
                                    #name_bin_field_label_as=None,
                                    compute_spatial_drift=True,
                                    x_coordinate_field="relative_x_center",
                                    y_coordinate_field="relative_y_center",
                                    spatial_strategy="equal",
                                    spatial_n_bins=10,
                                    )


#%%
drift_results = drift_suite_cls.drift_metrics()    



def concat_split_dfs(split_dfs: dict):
    df_list = []
    for split_nm, df in split_dfs.items():
        df["split_type"] = split_nm
        df_list.append(df)
        
    split_df = pd.concat(df_list)
    return split_df

full_split_df = concat_split_dfs(split_dfs=split_stats_res["split_dfs"])

object_dist_barplot = plot_bar(full_split_df.groupby(["split_type", "category_name"]).size().reset_index(name="count"),
                                    x="category_name", y="count",
                                    title="Category Distribution by Split",
                                    facet_row="split_type",
                                    color="split_type",
                                    color_discrete_sequence=px.colors.qualitative.Plotly,
                                    labels={"category_name": "Category", "count": "Count", "split_type": "Split"},
                                    text="count",
                                    height=500, width=700,
                                    
                                    )#.update_layout(showlegend=False, xaxis_tickangle=-45)
    
object_dist_barplot


drift_section = create_drift_section(drift_result=drift_results)


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

object_bias_content = generate_data_metric_section(metric_heading=f"Object Balance per Split",
                                                    fig=object_dist_barplot,
                                                    )



summary_kwargs = create_data_overview(split_dfs=split_stats_res["split_dfs"], 
                                          name="demo", id= str(uuid.uuid1())
                                          )


summary_section = create_section(kwargs=summary_kwargs)
authorship_section = create_section(kwargs=auth_kwargs)
data_collection_section = create_section(kwargs=data_col_kwargs)
labelling_section = create_section(kwargs=labelling_kwargs)
split_section = create_section(kwargs=split_kwargs)
transformation_section = create_section(kwargs=trans_kwargs)
license_section = create_section(kwargs=licence_kwargs)


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
                                        section=drift_scene_section, #'drift_contents
                                        )
motivation_section = create_section_head(header="Motivation and Intended Use",
                                            section=moti_content
                                            )
#%%
joined_section = "\n".join([title, 
                            authorship_section, 
                            motivation_section, 
                            data_overview,
                            data_collection_section, 
                            labelling_section, 
                            split_section,
                            wider_summary_statistics_section, 
                            dataspace_metrics, 
                            transformation_section,
                            license_section
                            ])

#%%
write_md_to_pdf(joined_section, pdf_path="dataset_card_join_raw.pdf")


def compute_stats_and_drift(object_stats_kwargs,
                            image_stats_kwargs,
                            metrics=metrics, field_to_bin=field_to_bin,
                            compute_spatial_drift=True,
                            x_coordinate_field="relative_x_center",
                            y_coordinate_field="relative_y_center",
                            spatial_strategy="equal",
                            spatial_n_bins=10,
                            make_date_card: bool = True,
                            **kwargs
                            ):
    split_stats_cls = SplitStats(object_stats_cls=ObjectStats,
                            image_loader_cls=ImageBatchDataset, 
                            object_stats_kwargs=object_stats_kwargs,
                            image_stats_kwargs=image_stats_kwargs
                            )  

    split_stats_res = split_stats_cls.compute_stats()
    
    distributions = split_stats_res["split_dfs"]
    drift_suite_cls = DriftMetricSuite(distributions=distributions,
                                        metrics=metrics, field_to_bin=field_to_bin,
                                        compute_spatial_drift=compute_spatial_drift,
                                        x_coordinate_field=x_coordinate_field,
                                        y_coordinate_field=y_coordinate_field,
                                        spatial_strategy=spatial_strategy,
                                        spatial_n_bins=spatial_n_bins,
                                        )

    drift_results = drift_suite_cls.drift_metrics() 
    
    if make_date_card:
        create_data_card(split_stats_result=split_stats_res,
                        drift_result=drift_results,
                        version_id=kwargs.get("version_id", str(uuid.uuid1())),
                        name=kwargs.get("name", "demo"),
                        intended_objects=kwargs.get("intended_objects"),
                        pdf_path=kwargs.get("pdf_path", PDF_PATH)
                        )
    return {"split_stats_result": split_stats_res,
            "drift_results": drift_results
            }


def create_data_card(split_stats_result,
                     drift_result,
                     version_id, name,
                     intended_objects=None,
                     pdf_path=PDF_PATH,
                     **kwargs
                    ):
    summary_stats_df = compute_summary_stats_wider(split_stats_result)
    wider_summary_table_fig = plot_summary_table(summary_stats_df, height=300)
    full_split_df = concat_split_dfs(split_dfs=split_stats_result["split_dfs"])

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
    moti_content = build_motivations_and_use(intended_tasks=kwargs.get("intended_tasks", intended_tasks),
                                            intended_objects=intended_objects, #["Person", "Helmet", "Safety vest"],
                                            )
    object_bias_content = generate_data_metric_section(metric_heading=f"Object Balance per Split",
                                                        fig=object_dist_barplot,
                                                        )



    summary_kwargs = create_data_overview(split_dfs=split_stats_res["split_dfs"], 
                                            name=name, id=version_id
                                            )


    summary_section = create_section(kwargs=summary_kwargs)
    authorship_section = create_section(kwargs=auth_kwargs)
    data_collection_section = create_section(kwargs=data_col_kwargs)
    labelling_section = create_section(kwargs=labelling_kwargs)
    split_section = create_section(kwargs=split_kwargs)
    transformation_section = create_section(kwargs=trans_kwargs)
    license_section = create_section(kwargs=licence_kwargs)


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
                                            section=drift_scene_section, #'drift_contents
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

    write_md_to_pdf(joined_section, pdf_path=pdf_path)


    
    
