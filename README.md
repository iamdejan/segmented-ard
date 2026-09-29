# Seg-ARD

Implementation of adversarial robust distillation for semantic segmentation task.

## Project Structure

- `train_teacher_normal.py` - Trains a U-Net (ResNet-18) teacher for BDD100k
  semantic segmentation with bias-correction sampling (weighted random sampler).
- `train_teacher_robust.py` - Trains a TRADES robust teacher using adversarial
  training (KL-divergence robustness term).
- `train_student_normal.py` - Distills the teacher into a smaller U-Net
  (MobileNet-V2) student via knowledge distillation (Dice+Focal hard loss + KL
  soft loss).
- `evaluate_performance.py` - Loads a trained checkpoint (default
  `./model/teacher.pt`) and reports the final segmentation metrics on the test
  split. It also renders a grid of the first test samples next to their
  predicted masks to `./model/predictions.png`.
- `evaluate_latency.py` - Loads a trained checkpoint (default
  `./model/teacher.pt`) and reports the per-image inference latency percentiles
  (P25, P50/median, mean, P75, P90, P95, P99) on the test split, in ms.

Each script reports the following metrics once at the end of training, using
the best checkpoint (chosen by lowest validation loss) evaluated on the test
set via `segmentation_models_pytorch` (`smp.metrics`):

- Pixel accuracy (`smp.metrics.accuracy`, `reduction="micro"`).
- Class-wise pixel accuracy (`smp.metrics.accuracy`, no reduction) — one
  accuracy value reported per class.
- Intersection over Union / IoU (`smp.metrics.iou_score`,
  `reduction="macro"`).
- Dice coefficient / F1 (`smp.metrics.f1_score`, `reduction="macro"`).

The `Unknown` class (id 19) is excluded from all metrics.
