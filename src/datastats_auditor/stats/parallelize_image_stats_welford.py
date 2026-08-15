
#%%
import numpy as np
from multiprocessing import Pool
from PIL import Image


_WORKER_PATHS = None
_WORKER_NORMALIZE = True

def _init_worker(paths, normalize):
    global _WORKER_PATHS, _WORKER_NORMALIZE
    _WORKER_PATHS = paths
    _WORKER_NORMALIZE = normalize

def _load_chw(path: str, normalize: bool):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr.transpose(2, 0, 1)  # CHW


def _merge_welford(a, b):
    """
    # -------------------------------------------------------------------
# Welford merge for two partial stats
# -------------------------------------------------------------------
    """
    if a["mean"] is None:
        return b
    if b["mean"] is None:
        return a

    mean_a, M2_a, n_a = a["mean"], a["M2"], a["count"]
    mean_b, M2_b, n_b = b["mean"], b["M2"], b["count"]

    delta = mean_b - mean_a
    n_total = n_a + n_b

    mean = mean_a + delta * (n_b / n_total)
    M2 = M2_a + M2_b + delta**2 * (n_a * n_b / n_total)

    return {
        "mean": mean,
        "M2": M2,
        "count": n_total,
        "min": np.minimum(a["min"], b["min"]),
        "max": np.maximum(a["max"], b["max"]),
        "heights": np.concatenate([a["heights"], b["heights"]]),
        "widths": np.concatenate([a["widths"], b["widths"]]),
    }



def _worker_welford(indices):
    """
    
    # -------------------------------------------------------------------
    # Worker: compute Welford stats for a shard of images
    # -------------------------------------------------------------------
    """
    mean = None
    M2 = None
    count = 0
    cmin = None
    cmax = None
    heights = []
    widths = []

    for idx in indices:
        path = _WORKER_PATHS[idx]
        try:
            img = _load_chw(path, _WORKER_NORMALIZE)
        except Exception:
            continue

        C, H, W = img.shape
        heights.append(H)
        widths.append(W)

        flat = img.reshape(C, -1)
        pixels = flat.shape[1]

        img_mean = flat.mean(axis=1)
        img_M2 = ((flat - img_mean[:, None])**2).sum(axis=1)
        img_min = flat.min(axis=1)
        img_max = flat.max(axis=1)

        if mean is None:
            mean = img_mean
            M2 = img_M2
            cmin = img_min
            cmax = img_max
            count = pixels
            continue

        delta = img_mean - mean
        new_total = count + pixels

        mean = mean + delta * (pixels / new_total)
        M2 = M2 + img_M2 + delta**2 * (count * pixels / new_total)
        count = new_total

        cmin = np.minimum(cmin, img_min)
        cmax = np.maximum(cmax, img_max)

    return {
        "mean": mean,
        "M2": M2,
        "count": count,
        "min": cmin,
        "max": cmax,
        "heights": np.array(heights, dtype=int),
        "widths": np.array(widths, dtype=int),
    }


def _merge_all_welford(partials):
    # deterministic left-to-right merge
    agg = partials[0]
    for p in partials[1:]:
        agg = _merge_welford(agg, p)
    return agg



def parallel_welford_stats(dataloader, num_workers=8):
    """
    # -------------------------------------------------------------------
# Main parallel stats function
# -------------------------------------------------------------------
    """
    paths = [str(p) for p in dataloader.paths]
    N = len(paths)
    if N == 0:
        raise ValueError("No images found")

    workers = min(num_workers, N)
    shards = np.array_split(np.arange(N), workers)
    shards = [s.tolist() for s in shards if len(s) > 0]

    with Pool(processes=len(shards), initializer=_init_worker,
              initargs=(paths, dataloader.normalize)) as pool:
        partials = pool.map(_worker_welford, shards)

    agg = {
        "mean": None, "M2": None, "count": 0,
        "min": None, "max": None,
        "heights": np.array([], dtype=int),
        "widths": np.array([], dtype=int),
    }

    #for p in partials:
        #agg = _merge_welford(agg, p)
    agg = _merge_all_welford(partials)


    var = agg["M2"] / agg["count"]
    std = np.sqrt(var)

    return {
        "mean": agg["mean"],
        "var": var,
        "std": std,
        "min": agg["min"],
        "max": agg["max"],
        "height_stats": {
            "min": int(agg["heights"].min()),
            "max": int(agg["heights"].max()),
            "mean": float(agg["heights"].mean()),
        },
        "width_stats": {
            "min": int(agg["widths"].min()),
            "max": int(agg["widths"].max()),
            "mean": float(agg["widths"].mean()),
        },
    }




class ParallelImageStats:
    def __init__(self, image_loader, num_workers=8):
        self.image_loader = image_loader
        self.num_workers = num_workers

    def compute_image_stats(self):
        stats = parallel_welford_stats(self.image_loader, self.num_workers)

        # ensure float32 output
        for k in ["mean", "var", "std", "min", "max"]:
            stats[k] = np.asarray(stats[k], dtype=np.float32)

        return stats


#%%

#image_loader = ImageBatchDataset(image_dir=train_imgdir)

#%%
#welford_imgstat_cls = ParallelImageStats(image_loader=image_loader, num_workers=num_processors-2)
# %%
#welford_res = welford_imgstat_cls.compute_image_stats()


# %%
























