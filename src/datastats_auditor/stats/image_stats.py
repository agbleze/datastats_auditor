



#%%
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import os

#%%   
class ExampleDataset(Dataset):
    def __init__(self, num_samples=100, 
                    num_channels=12,
                    height=512, width=512
                    ):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, num_channels, height, width)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

#%%
example_dataloader = DataLoader(ExampleDataset(), 
                            batch_size=10, 
                            shuffle=False
                            )


#%%

img = np.random.randint(low=0, high=256, size=(512, 512, 3), dtype=np.uint8)


#%%
Image.fromarray(img).show()
#%%

img.min(axis=(0,1))


#%%
from glob import glob
from pathlib import Path

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", "webp"}

class ImageBatchDataset:
    def __init__(self, image_dir, batch_size=10, normalize=True):
        self.root = Path(image_dir)
        self.paths = [p for p in self.root.glob("*") if p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        self.num_samples = len(self.paths)
        self.batch_size = batch_size
        self.normalize = normalize
        
    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            batch_paths = self.paths[i:i+self.batch_size]
            batch = []
            for path in batch_paths:
                try:
                    img = np.array(Image.open(path), dtype=np.float32)
                except Exception as e:
                    raise RuntimeError(f"Error loading image {path}: {e}")
                if self.normalize:
                    img = img / 255.0
                    #img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
                if img.ndim == 2:
                    img = img[..., None] # grayscale to (H,W,1)
                img = np.transpose(img, (2,0,1)) # HWC -> CHW
                batch.append(img)
            yield np.stack(batch, axis=0) # (B,C,H,W)
            
    def __getitem__(self, idx):
        return self.paths[idx]
        


#%%
imgdir = "/home/lin/codebase/chili_seg/Chili_seg.v3i.coco-segmentation/train"

#%%
dataset = ImageBatchDataset(imgdir, batch_size=10, normalize=True)

#%%

len(dataset)

#%%

dataset[0]
#%%
list(Path(imgdir).glob("*"))

#%%


import numpy as np

def compute_dataset_stats(dataloader):
    total_pixels = 0
    mean = None
    M2 = None
    cmin = None
    cmax = None

    heights = []
    widths = []

    for batch in dataloader:  # batch is a list of images (C,H,W)
        for img in batch:
            C, H, W = img.shape
            heights.append(H)
            widths.append(W)

            flat = img.reshape(C, -1).T  # (pixels, C)
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


#%%
data = [
    np.random.randn(3, 32, 32),
    np.random.randn(3, 64, 64),
    np.random.randn(3, 48, 80),
    np.random.randn(3, 32, 100),
]

class Loader:
    def __iter__(self):
        yield data[:2]   # batch 1
        yield data[2:]   # batch 2

result = compute_dataset_stats(Loader())

# Ground truth
flat = np.concatenate([img.reshape(3, -1).T for img in data], axis=0)
print("Streaming mean:", result["mean"])
print("True mean:", flat.mean(axis=0))
print("Streaming var:", result["var"])
print("True var:", flat.var(axis=0))
print("Heights:", result["height_stats"])
print("Widths:", result["width_stats"])
print("streaming std", result["std"])
print("True std", flat.std(axis=0))

#%%

loaded_data = Loader()

#%%

type(loaded_data)

fetched_data = next(iter(loaded_data))

#%%

fetched_data[1][2].shape


#%%

import numpy as np

def global_compute_dataset_stats(dataloader):
    global_sum = None
    global_sum_sq = None
    global_min = None
    global_max = None
    global_pixels = 0

    heights = []
    widths = []

    for batch in dataloader:  # batch is a list of images (C,H,W)
        for img in batch:
            C, H, W = img.shape
            heights.append(H)
            widths.append(W)

            flat = img.reshape(C, -1)  # (C, pixels)
            pixels = flat.shape[1]

            img_sum = flat.sum(axis=1)        # (C,)
            img_sum_sq = (flat**2).sum(axis=1)
            img_min = flat.min(axis=1)
            img_max = flat.max(axis=1)

            if global_sum is None:
                global_sum = img_sum
                global_sum_sq = img_sum_sq
                global_min = img_min
                global_max = img_max
                global_pixels = pixels
                continue

            global_sum += img_sum
            global_sum_sq += img_sum_sq
            global_min = np.minimum(global_min, img_min)
            global_max = np.maximum(global_max, img_max)
            global_pixels += pixels

    mean = global_sum / global_pixels
    var = global_sum_sq / global_pixels - mean**2

    return {
        "mean": mean,
        "var": var,
        "min": global_min,
        "max": global_max,
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
    
#%%

result_global = global_compute_dataset_stats(Loader())
print("Global mean:", result_global["mean"])
print("Global var:", result_global["var"])  


flat = np.concatenate([img.reshape(3, -1).T for img in data], axis=0) 
print("Streaming mean:", result_global["mean"]) 
print("True mean:", flat.mean(axis=0)) 
print("Streaming var:", result_global["var"]) 
print("True var:", flat.var(axis=0))
print("Streaming std:", result_global["std"])
print("True std:", flat.std(axis=0))


#%%

stats = compute_dataset_stats(dataloader=dataset)

#%%
stats

#%%

import psutil
print("Memory usage (GB):", psutil.Process().memory_info().rss / 1e9)


def get_memory_info():
    mem = psutil.virtual_memory()
    return {"total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9,
            "free_gb": mem.free / 1e9,
            "percent": mem.percent,
            "active_gb": mem.active / 1e9,
            "inactive_gb": mem.inactive / 1e9,
            }
    
    
def estimate_image_memory_size_GB(img):
    print(f"Actual memory usage for this image: {img.nbytes / 1e9:.2f} GB")
    return img.nbytes / 1e9


def should_load_full_dataset_into_memory(estimated_memory_gb, available_memory_gb, buffer=0.5):
    return estimated_memory_gb < (available_memory_gb * (1 - buffer))


def choose_batch_size(avg_dataset_memory_gb, available_memory_gb, buffer=0.5):
    if buffer > 1.0:
        raise ValueError(f"buffer {buffer} cannot be larger than available memory {available_memory_gb}")
    
    max_memory = available_memory_gb * (1 - buffer)
    batch_size = int(max_memory // avg_dataset_memory_gb)
    return max(1, batch_size)
    

#%%

get_memory_info()


#%%
img = next(iter(dataset))#[0]


#%%

estimate_image_memory_size_GB(img)

#%%

img.mean(axis=(0,2,3))

#%%

len(dataset)

#%%

full_dataset = ImageBatchDataset(imgdir, batch_size=len(dataset), normalize=True)   


#%%

all_imgs = next(iter(full_dataset))

#%%
all_imgs.mean(axis=(0,2,3))


#%%

all_imgs.shape
#%%

compute_dataset_stats(full_dataset)


#%%

all_batches = list(full_dataset)

#%%
_all_imgs = np.concatenate(all_batches, axis=1)


#%%

_all_imgs.mean(axis=(0,2,3))

#%%

"""
Docstring for datastats_auditor.src.datastats_auditor.stats.image_stats

Per channel min, max, mean, stddev for all images in the dataset. Useful for sanity checking and normalization.


"""



