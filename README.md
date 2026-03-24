# Resolution-Agnostic Knowledge Distillation for Remote Sensing Object Detection

End-to-end pipeline: knowledge distillation from a high-capacity teacher (Swin-T + FPN) trained on high-res DOTA images to a lightweight student (MobileNetV2 + FPN) operating on 128×128 low-res inputs, with intermediate feature-map alignment via spatial projection adapters.

---

## Compute Requirements

| Phase | GPU VRAM | Time (500 images, 50 epochs) | Can use CPU? |
|---|---|---|---|
| Teacher training (1024×1024, Swin-T+FPN, bs=4) | 10–12 GB (fp32) / 6 GB (fp16) | ~3–4 hours on RTX 3060+ | No |
| Student baseline (128×128, MobileNetV2+FPN, bs=16) | 3–4 GB | ~30–45 min | Painful but possible |
| Student KD (128×128, frozen teacher + student, bs=8) | 8–10 GB (fp32) / 5 GB (fp16) | ~2–3 hours | No |
| Evaluation + Visualization | 2–3 GB | ~5–10 min | Yes |

> [!TIP]
> **Minimum recommended**: Single GPU with 8 GB VRAM (e.g., RTX 3060, RTX 4060, T4). Use fp16 mixed precision throughout. If you have Google Colab Pro (A100/V100), everything runs comfortably.

> [!NOTE]
> **Disk space**: ~2–3 GB for DOTA subset images + ~500 MB for model checkpoints + ~200 MB for cached features. Total ~4 GB.

---

## Dataset Strategy

| Role | Source | Resolution | Purpose |
|---|---|---|---|
| Teacher HR | DOTA v1.0 original | 1024×1024 crops | Teacher trains on HR patches with fine spatial detail |
| Student LR | DOTA 512×512 version | 128×128 (downsampled 4×) | Student trains on degraded versions |
| Cross-dataset eval | NWPU VHR-10 | Resized to 128×128 | Tests generalization |

**Resolution gap**: 1024 → 128 = 8× downsample.

**DOTA classes used** (15 total): plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│ TEACHER (frozen during KD)                       │
│ Input: 1024×1024 HR image                        │
│ Swin-T Backbone → FPN → RetinaNet Head           │
│   ├── Stage 1 features (F_t1)                    │
│   ├── Stage 2 features (F_t2)                    │
│   ├── Stage 3 features (F_t3)                    │
│   └── Detection logits (cls + bbox regression)   │
└──────────────────────┬───────────────────────────┘
                       │ Feature KD + Logit KD
                       ▼
┌──────────────────────────────────────────────────┐
│ STUDENT (trained with KD)                        │
│ Input: 128×128 LR image                          │
│ MobileNetV2 Backbone → Lightweight FPN → Head    │
│   ├── Adapter(F_s1) → align to F_t1              │
│   ├── Adapter(F_s2) → align to F_t2              │
│   ├── Adapter(F_s3) → align to F_t3              │
│   └── Detection logits                           │
└──────────────────────────────────────────────────┘

Total Loss = α·L_detection + β·L_logit_kd + γ·L_feature_kd
```

---

## Project Structure

```
Project/
├── configs/
│   └── config.yaml              # All hyperparameters
├── data/
│   ├── dataset.py               # DOTA detection dataset (HR + LR)
│   └── prepare_data.py          # Crop DOTA images, create splits
├── models/
│   ├── teacher.py               # Swin-T + FPN + RetinaNet head
│   ├── student.py               # MobileNetV2 + FPN + detection head
│   ├── fpn.py                   # Feature Pyramid Network (shared)
│   ├── detection_head.py        # RetinaNet-style cls + box head
│   └── adapters.py              # Spatial projection adapters
├── losses/
│   ├── detection_loss.py        # Focal loss + Smooth L1 (det)
│   └── distillation.py          # Logit KD + Feature KD losses
├── train_teacher.py             # Phase 1: train teacher on HR
├── train_student_baseline.py    # Phase 2a: student without KD
├── train_student_kd.py          # Phase 2b: student with KD
├── evaluate.py                  # mAP, F1, FPS, model size
├── visualize.py                 # Det boxes, heatmaps, features
├── utils.py                     # Anchors, NMS, metrics, helpers
├── requirements.txt
└── README.md
```

---

## Verification Plan

### Smoke Tests

```bash
# Verify dataset loads correctly with boxes
python -c "from data.dataset import DOTADetectionDataset; ds = DOTADetectionDataset('data/processed', 'train', 1024); img, tgt = ds[0]; print(f'Image: {img.shape}, Boxes: {tgt[\"boxes\"].shape}')"

# Verify teacher forward pass
python -c "import torch; from models.teacher import TeacherDetector; m = TeacherDetector(15); out = m(torch.randn(1,3,1024,1024)); print('Teacher OK')"

# Verify student forward pass
python -c "import torch; from models.student import StudentDetector; m = StudentDetector(15); out = m(torch.randn(1,3,128,128)); print('Student OK')"

# Verify KD loss computes
python -c "import torch; from losses.distillation import DistillationLoss; print('Loss OK')"
```

### End-to-End

```bash
python train_teacher.py --epochs 2 --subset 20
python train_student_baseline.py --epochs 2 --subset 20
python train_student_kd.py --epochs 2 --subset 20
python evaluate.py
python visualize.py
```

### Expected Result Pattern

| Model | mAP@0.5 | FPS | Params | Size |
|---|---|---|---|---|
| Teacher (Swin-T, 1024) | High (~0.65-0.75) | Slow (~5-10) | ~30M | ~120MB |
| Student baseline (MobileNetV2, 128) | Low (~0.30-0.40) | Fast (~50-80) | ~3M | ~12MB |
| Student KD (ours, 128) | Mid (~0.50-0.60) | Fast (~45-70) | ~3.5M | ~14MB |
