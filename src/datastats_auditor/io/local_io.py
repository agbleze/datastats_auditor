from pathlib import Path
from datastats_auditor.stats.io.baseio import IterableDataset
from datastats_auditor.stats.constant import VALID_IMAGE_EXTENSIONS
import numpy as np
from PIL import Image


class ImageBatchDataset(IterableDataset):
    name = "local_image_loader"
    status = "stable"
    
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
                if img.ndim == 2:
                    img = img[..., None] # grayscale to (H,W,1)
                img = np.transpose(img, (2,0,1)) # HWC -> CHW
                batch.append(img)
            yield np.stack(batch, axis=0) # (B,C,H,W)
            
    def __getitem__(self, idx):
        return self.paths[idx]
        