from shapely.geometry import box
from shapely.ops import unary_union



def compute_foreground_area_union(df):
    def bbox_area_union(group):
        polys = []
        for _, row in group.iterrows():
            x_min = row["bbox_x"]
            y_min = row["bbox_y"]
            x_max = row["bbox_x"] + row["bbox_w"]
            y_max = row["bbox_y"] + row["bbox_h"]
            polys.append(box(x_min, y_min, x_max, y_max))

        if not polys:
            return 0.0

        union_poly = unary_union(polys)
        return union_poly.area 
    ratios = (df.groupby("image_id")
                .apply(bbox_area_union)
                .rename("foreground_union_area_per_image")
                .reset_index()
                )

    return df.merge(ratios, on="image_id")
