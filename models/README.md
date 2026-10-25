# Model weights

Weights are not stored in this repository. They are hosted on Hugging Face:
**https://huggingface.co/wahyunan/sidewalk-obstacle-geolocation**

`notebooks/03_geolocate_obstacles.ipynb` downloads them automatically with `hf_hub_download`.

| Model | File | Notes |
|---|---|---|
| SegFormer-B0 (sidewalk) | [`segformer_final.ckpt`](https://huggingface.co/wahyunan/sidewalk-obstacle-geolocation/blob/main/segformer_final.ckpt) | PyTorch Lightning checkpoint, last epoch (20). Used for all results in the README |
| YOLO11s (13 obstacle classes) | [`yolo-obstacle-detection.pt`](https://huggingface.co/wahyunan/sidewalk-obstacle-geolocation/blob/main/yolo-obstacle-detection.pt) | Ultralytics `best.pt` (epoch 28) |
| Depth Pro | loaded automatically from `apple/DepthPro-hf` | Not redistributed |

```python
from huggingface_hub import hf_hub_download
seg_ckpt = hf_hub_download("wahyunan/sidewalk-obstacle-geolocation", "segformer_final.ckpt")
yolo_pt  = hf_hub_download("wahyunan/sidewalk-obstacle-geolocation", "yolo-obstacle-detection.pt")
```

The SegFormer checkpoint must be loaded with `transformers==4.57.6`
(SegFormer layer names changed in transformers 5.x).
