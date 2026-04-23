# Week 7 – Ball Action Spotting (Temporal Aggregation)

This week extends the ball action spotting task from Week 6 by introducing **temporal aggregation** via a UNet-like architecture built on top of the X3D-M spatiotemporal backbone. The main goals are:

- Implement intermediate temporal reduction (L → L' → L) using a UNet-like encoder-decoder.
- Evaluate model performance at two temporal tolerances: **1.0s** and **0.5s**.
- Refine the best model from Week 6 using insights from the experimental pipeline.

---

## 📁 Repository Structure

```
week7/
├── config/                  # JSON config files for each experiment
├── dataset/
│   ├── datasets.py          # Dataset loader
│   └── frame.py             # ActionSpotDataset, ActionSpotVideoDataset, TGLS
├── model/
│   ├── model_spotting.py    # All architectures: X3D base, +GRU, +UNet, +UNet+GRU
│   ├── model_spotting_w6.py # W6 model kept for qualitative comparison
│   └── modules.py
├── util/
│   ├── eval_spotting.py     # Dual-tolerance evaluation (1s and 0.5s) + NMS
│   └── io.py
├── main_spotting.py         # Training and evaluation script
└── qualitative_spotting.py  # Animated qualitative comparison (W6 vs W7)
```

---

## 🏗️ Best Model Architecture

**X3D-M Encoder → UNet Decoder → Linear Projection → Bi-GRU → FC Head**

The X3D-M backbone processes the full clip spatiotemporally. After spatial pooling, a **MaxPool1d bottleneck** compresses the temporal dimension (T=50 → T'=25). The decoder recovers it via **ConvTranspose1d with skip connections** from each encoder stage. A **linear projection** (64 → 192 dims) aligns the decoder output with the X3D-M embedding space before passing it to the same **Bi-GRU** used in Exp 2, enabling a fair comparison.

---

## 🧪 Results

| Model | mAP10 @1s | mAP10 @0.5s |
|-------|-----------|-------------|
| W6 Best (RNY008 + TCN + NMS) | 35.5 | 30.2 |
| Exp 1: X3D-M base | 41.3 | 36.3 |
| Exp 2: X3D-M + Bi-GRU | 45.3 | 40.0 |
| Exp 3: X3D-M + UNet | 39.1 | 34.8 |
| Exp 4: X3D-M + UNet + Bi-GRU | 43.7 | 40.1 |
| Exp 5–7: Stride & Clip Length search | up to 45.3 | — |
| Exp 8: AvgPool bottleneck | 45.8 | 40.8 |
| **Best: UNet + Bi-GRU (Max, 2L, 512h)** | **47.2** | **41.9** |
| + TGLS | 42.7 | 39.4 |
| + Optimized NMS | 43.2 | 38.9 |

---

## ⚙️ Running the Code

```bash
pip install -r requirements.txt
pip install pytorchvideo
```

On first run, set `store_mode: "store"` in the config to generate clip caches. Then switch to `"load"` for subsequent runs.

```bash
python main_spotting.py --model <config_name>
```

Results are logged to **WandB** and saved locally as CSV files under `save_dir/<model_name>/`.

---

## 📦 Checkpoint

Best model checkpoint available at:

> 🔗 [Best Model W7 – Google Drive](https://drive.google.com/drive/folders/1b0Fe95kcJf9bDk4NaBILYh-AnUrjxH9H?usp=drive_link)

