from datastats_auditor import datastats_auditor












# def compute_foreground_ratio_union(df):
#     def per_image_union(group):
#         polys = []
#         for _, row in group.iterrows():
#             x_min = row["bbox_x"]
#             y_min = row["bbox_y"]
#             x_max = row["bbox_x"] + row["bbox_w"]
#             y_max = row["bbox_y"] + row["bbox_h"]
#             polys.append(box(x_min, y_min, x_max, y_max))

#         if not polys:
#             return 0.0

#         union_poly = unary_union(polys)
#         img_area = group["image_width"].iloc[0] * group["image_height"].iloc[0]
#         return union_poly.area / img_area

#     ratios = (
#         df.groupby("image_id")
#           .apply(per_image_union)
#           .rename("foreground_ratio_union")
#           .reset_index()
#     )

#     return df.merge(ratios, on="image_id")


    

#%%

#trn_df = compute_foreground_ratio_union(train_df)

#%%

trn_df[["foreground_ratio_union", "foreground_ratio", "bbox_area_norm",
        "occupancy_per_image"
        ]]

#%%

trn_df[trn_df["foreground_ratio_union"] > trn_df["occupancy_per_image"]]



#%%


num_bboxes = trn_df.groupby("image_id").size().rename("num_bboxes").reset_index()#["bbox"].count()

td = trn_df.merge(num_bboxes, on="image_id", how="left")


#%%

td.columns
#%%

td["image_id"].max()
#pd.concat()
td[td["image_id"]==2100][["image_id","num_bboxes"]]
#%%
num_bboxes["image_id"].max()
num_bboxes[num_bboxes["image_id"]==2100][["image_id","num_bboxes"]]

#%%

trn_df.columns

#%%

td.sort_values(by="image_id")[["image_id", "num_bboxes"]]

#%%

num_bboxes.sort_values(by="image_id")
#%%


#trn_df.drop("num_bboxes", axis=1, inplace=True)

td["bbox_area_norm"].var()

td.groupby("image_id")["bbox_area_norm"].var().fillna(0)
#%%

import pkg_resources

packages = {dist.project_name: dist.version for dist in pkg_resources.working_set}

# %%
packages
# %%

import inspect
from functools import wraps

def capture_params(store_attr="_captured_params"):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Store parameters on the function object
            setattr(wrapper, store_attr, dict(bound.arguments))

            return func(*args, **kwargs)

        return wrapper
    return decorator

@capture_params()
def preprocess_images(images, resize=(640, 640), normalize=True):
    # ... preprocessing logic ...
    return images


preprocess_images(data, resize=(512, 512), normalize=False)

print(preprocess_images._captured_params)


##

PIPELINE_LOG = []

def capture_step(name=None):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            PIPELINE_LOG.append({
                "step": name or func.__name__,
                "params": dict(bound.arguments)
            })

            return func(*args, **kwargs)

        return wrapper
    return decorator


@capture_step("deduplication_exact")
def dedupe_exact(images, hash_type="chash", hash_size=8):
    return images

@capture_step("splitter")
def split_data(images, train=0.8, val=0.2, seed=2023):
    return images

dedupe_exact(data, hash_size=16)
split_data(data, train=0.7, val=0.3)


from pprint import pprint
pprint(PIPELINE_LOG)


Capture Return Values Too
def capture_params_and_output(store_attr="_captured"):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            result = func(*args, **kwargs)

            setattr(wrapper, store_attr, {
                "params": dict(bound.arguments),
                "output": result
            })

            return result

        return wrapper
    return decorator



def build_preprocessing_section():
    lines = []
    for step in PIPELINE_LOG:
        lines.append(f"### {step['step']}")
        for k, v in step["params"].items():
            if k != "images":
                lines.append(f"- **{k}:** {v}")
    return "\n".join(lines)
