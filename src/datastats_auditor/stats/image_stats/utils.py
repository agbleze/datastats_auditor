import numpy as np




def compute_dataset_stats(dataloader):
    total_pixels = 0
    mean = None
    M2 = None
    cmin = None
    cmax = None

    heights = []
    widths = []

    for batch in dataloader:
        for img in batch:
            C, H, W = img.shape
            heights.append(H)
            widths.append(W)

            flat = img.reshape(C, -1).T
            pixels = flat.shape[0]

            img_mean = flat.mean(axis=0)
            img_M2 = ((flat - img_mean)**2).sum(axis=0)
            img_min = flat.min(axis=0)
            img_max = flat.max(axis=0)

            if mean is None:
                mean = img_mean
                M2 = img_M2
                cmin = img_min
                cmax = img_max
                total_pixels = pixels
                continue

            delta = img_mean - mean
            new_total = total_pixels + pixels

            mean += delta * (pixels / new_total)
            M2 += img_M2 + delta**2 * total_pixels * pixels / new_total
            total_pixels = new_total

            cmin = np.minimum(cmin, img_min)
            cmax = np.maximum(cmax, img_max)

    var = M2 / total_pixels

    return {
        "mean": mean,
        "var": var,
        "min": cmin,
        "max": cmax,
        "std": np.sqrt(var),
        "height_stats": {
            "min": min(heights),
            "max": max(heights),
            "mean": np.mean(heights)
        },
        "width_stats": {
            "min": min(widths),
            "max": max(widths),
            "mean": np.mean(widths)
        }
    }

