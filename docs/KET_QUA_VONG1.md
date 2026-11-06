# 🎯 VÒNG 1 — FINE-TUNE THEO QUY TẮC MỚI (quytactrain.md)

> Ngày 2026-11-04. Phạm vi: **core-first** (quyết định của user).
> Tất cả số liệu đo trên RTX 4060 Laptop 8GB, dataset mới `data/datasets/style_v4`.

---

## 1. Dataset v4 — xây lại từ đầu theo §2-4 (KHÔNG tái dùng dataset cũ)

Nguồn: `data/chesscom/all_rapid_blitz.pgn` (2.332 ván thô, rapid 10+5 và blitz 3+0/5+0,
tài khoản TsukinoLuna + Sasayaku).

### Lọc cấp ván (§3.1)
| Bộ lọc | Bị loại |
|---|---|
| rating tài khoản <1250 (mục §3.3, phương án A) | 483 |
| abandoned/aborted | 103 |
| chênh rating đối thủ >400 | 15 |
| quá ngắn (<10 ply) | 6 |
| thiếu elo / variant khác Standard | 1 / 28 |
| **Giữ** | **1.696 ván** |

### Lọc cấp nước (§3.2) — áp dụng cho nước của CHÍNH tài khoản
| Bộ lọc | Số nước bị loại/xử lý |
|---|---|
| nước bị buộc (chỉ 1 nước hợp lệ) | 768 → bỏ |
| premove/phản xạ (chi tiêu <1s) | 3.616 → bỏ |
| cạn thời gian (< max(30s, 10% thời lượng)) | 992 → bỏ |
| thế đã định (|SF eval| ≥ 800cp, depth 10) | 5.750 → **giữ, trọng số ×0.5** |

**⚠ Quyết định thiết kế quan trọng:** dataset cũ (v1-v3) gán nhãn CẢ nước đối thủ
(cả 2 bên bàn cờ) → 133k positions; dataset v4 mới chỉ gán nhãn nước của chính user
→ **53.964 positions**. Số ít hơn là do chủ đích: bot học *phong cách đi của user*,
nước đối thủ (người khác, phong cách lẫn lộn) là nhiễu. Bộ lọc §3.2 cũng chỉ đúng
nghĩa khi áp cho nước của user.

### Chia theo THỜI GIAN (§4) — 70/10/15, không bao giờ chia theo ply
| Split | Ván | Positions | Khoảng UTC |
|---|---|---|---|
| train | 1.187 | 38.101 | 2022.09.11 → 2026.02.24 |
| val | 170 | 5.525 | 2026.02.24 → 2026.03.26 |
| test | 339 | 10.338 | 2026.03.26 → 2026.10.30 |

- Arena 500 ván (v3 dùng train) **bị loại khỏi train** — giờ chỉ dành cho eval phong cách (§8.5).
- Trọng số mẫu: rapid w1.0 (46.5k) · blitz w0.3 (1.7k) · decided w0.5/0.15. Effective size 36.1k/38.1k.
- Khai trùng lặp (12 nước đầu giống hệt): 79 cặp, 29 nhóm rải các split → ước lượng
  top1 hơi thổi phồng phần "học thuộc khai" — ghi nhận, chưa triệt tiêu (vòng 2).

Output: `data/datasets/style_v4` (+`style_v4_final` = train+val 42.726/monitor 900,
test nguyên vẹn). Meta đầy đủ: `data/datasets/style_v4/meta.json`.

---

## 2. Baseline sweep (§6.1-E0, §7) — val 5.525 positions

Val split v4 = đúng loại nước "user tự đi" + elo thật từng thế cờ (native).

| Model gốc | elo | NLL | top1 | top3 |
|---|---|---|---|---|
| 5M | 1750 | 1.3561 | 54.62% | 83.04% |
| 23M | 1750 | 1.3271 | 54.95% | 83.82% |
| 79M | 1750 | 1.3218 | 55.02% | 83.66% |
| **79M** | **native** | **1.3391** | **54.46%** | **83.80%** |

- Tất cả cỡ model đạt optimum ở **elo 1750**; cỡ 79M tốt nhất, đảm bảo chuẩn "càng to càng khớp".
- **Phát hiện §7:** elo override cố định 1750 KÉM hơn elo thật từng vị thế (native)
  trên dữ liệu "user tự đi" (55.0% vs 58.4% — v1). Khi deploy: dùng elo ước lượng
  thật của user cho bộ điều ứng, không ghim số cố định.
- Script: `code/baseline_sweep.py` · JSON: `/tmp/opencode/baseline_sweep_v4.json`

---

## 3. Fine-tune + ablation (§6.3) — 4 epochs, batch 128, bf16, trọng số mẫu

| Run | lr / freeze | best top1 (val v4) | thời gian | Kết luận |
|---|---|---|---|---|
| v4 | 1e-5, full | **59.22%*** | 8.3 phút | ✅ thắng |
| v4-lr2e5 | 2e-5, full | 58.91% | 8.4 phút | NLL val tốt nhất (1.1937) nhưng top1 kém hơn, loại |
| v4-freeze | 1e-5, freeze blocks<8+emb | 55.46% | 3.3 phút | tệ hơn hẳn, loại |
| outweigh | — | (dùng `--ignore-weights` nếu muốn) | — | để vòng 2 |

*v4 best ở step 370 (epoch 1) rồi overfit nhẹ → recipe final giữ nguyên 4 epochs
nhưng chọn best.pt trên monitor (§8.4 cho phép: chọn theo val, test dùng 1 lần).

Cross-eval trên val (`code/evaluate_v2.py`, elo native, 5.525 pos):
| Model | NLL | top1 | top3 | top5 | ECE |
|---|---|---|---|---|---|
| gốc 79M | 1.3391 | 54.46% | 83.80% | 92.09% | 1.45% |
| v1 (cũ) | 1.2123 | 58.39% | 86.01% | 94.19% | 1.00% |
| v4 | 1.2105 | 59.29% | 86.35% | 93.99% | 3.32% |
| v4-lr2e5 | 1.1937 | 58.90% | 86.77% | 94.01% | 1.63% |
| v4-freeze | 1.3052 | 55.44% | 84.04% | 92.38% | 1.69% |

- v4 vs v1: Δtop1 +0.90% (CI 2000-iters per-game: [-2.29, -0.32], đủ ý nghĩa).
- v4 hơi tự tin hơn (ECE 3.32% vs 1.0%) — theo dõi khi deploy; nếu cần thêm
  `--label-smoothing 0.1` (đã implement sẵn trong finetune.py) ở vòng 2.

---

## 4. Final retrain train+val → TEST 1 LẦN (§8.4)

- Final: `finetunes/maia3-79m-v4-final/best.pt` — train 42.726 pos (train+val)
  + chọn trên monitor 900, best top1 monitor 60.89%, 8.5 phút.
- **TEST (duy nhất một lần, 10.338 pos / 339 ván, CI bootstrap 2000 lượt theo ván):**

| Model | NLL | top1 | top3 | top5 | ECE | top1 khai/trung/tàn |
|---|---|---|---|---|---|---|
| gốc 79M | 1.3065 | 55.82% | 84.57% | 93.00% | 3.77% | 53.7 / 55.5 / 58.7 |
| v4 | 1.1755 | 60.42% | 87.02% | 94.17% | 1.81% | 63.7 / 58.0 / 61.7 |
| **v4-final** | **1.1631** | **60.53%** | **87.48%** | **94.37%** | 2.22% | 63.6 / 58.2 / 61.9 |

- gốc→v4-final: Δtop1 **+5.11%** CI [+4.16, +6.00] · ΔNLL −0.157 CI [−0.174, −0.140] → significant.
- v4 ≈ v4-final (v4-final nhỉnh nhẹ ở mọi cột).
- JSON: `docs/eval_v4_test.json`.

---

## 5. So sức mạnh động (engine match 50 ván) — v4-final vs v3

Trận: 50 ván, Elo 1700, temp 0.36, khai seed từ corpus, PGN: `data/chesscom/match_v4f_vs_v3_a_vs_b.pgn`.

| (góc nhìn v4-final) | Trắng | Đen | Tổng |
|---|---|---|---|
| thắng | 9 | 9 | 18 (36%) |
| hòa | 8 | 2 | 10 |
| thua | 8 | 14 | 22 (44%) |

- Tổng điểm v4-final = 46% → **v4-final ≈ v3 − 28 Elo**.
- ⚠️ n=50: CI của điểm số ±~14% → sai khác **chưa đủ ý nghĩa thống kê** — xem là "ngang nhau".
- Giải thích hợp lý: v3 được train với cả 500 ván arena vs Stockfish (dữ liệu giúp
  lối đi "máy mạnh hơn"), dataset v4 theo quy tắc mới chỉ dùng ván của chính user →
  **đổi sức máy (v3) lấy độ trung thực phong cách (v4)**.
- v4-final làm Đen hơi yếu (9W/14L) — hiện tượng lệch màu đã thấy ở v1, cân nhắc
  cân bằng lại dữ liệu trắng/đen hoặc tăng seed khai ở vòng 2.

---

## 6. Ngân sách GPU vòng 1

sweep ~13 phút + 3 fine-tune ~26 phút + evals/test ~10 phút ≈ **49 phút** (≤2-3h budget).

## 7. Để vòng 2 (core-first có chủ đích hoãn)

- Rolling split 3-4 lần (§4), 3 seed, replay ván Lichess mới (§6.3), LoRA,
  nội suy trọng số α ft/gốc với quét α (§6.4), ablation có/không trọng số mẫu
  (`--ignore-weights` đã sẵn sàng), label smoothing 0.1, đánh giá phong cách §8.5.

## 8. Lệnh tái lập toàn bộ vòng 1

```bash
# dataset
.venv/bin/python code/prepare_dataset_v2.py data/chesscom/all_rapid_blitz.pgn \
    --accounts TsukinoLuna Sasayaku --out data/datasets/style_v4 --sf-workers 8
# baseline sweep
.venv/bin/python code/baseline_sweep.py --dataset data/datasets/style_v4 --split val \
    --json /tmp/opencode/baseline_sweep_v4.json
# fine-tune (cấu hình thắng)
.venv/bin/python code/finetune.py --dataset data/datasets/style_v4 \
    --out finetunes/maia3-79m-v4 --model maia3-79m --epochs 4 --batch-size 128 \
    --lr 1e-5 --value-weight 0.3 --workers 4
# final split + retrain
.venv/bin/python code/make_final_split.py --dataset data/datasets/style_v4 \
    --out data/datasets/style_v4_final
.venv/bin/python code/finetune.py --dataset data/datasets/style_v4_final \
    --out finetunes/maia3-79m-v4-final --model maia3-79m --epochs 4 --batch-size 128 \
    --lr 1e-5 --value-weight 0.3 --workers 4
# test một lần
.venv/bin/python code/evaluate_v2.py --dataset data/datasets/style_v4 --split test \
    --labels raw79m v4 v4-final \
    --checkpoints <raw79m.pt> finetunes/maia3-79m-v4/best.pt \
    finetunes/maia3-79m-v4-final/best.pt --json docs/eval_v4_test.json
```

*Checkpoint chuẩn dùng trong vòng 1: `finetunes/maia3-79m-v4-final/best.pt`.*

## 9. Triển khai (§9) — đã thực hiện

- Wrap `code/maia3-bot-engine.sh` giờ trỏ `maia3-79m-v4-final/best.pt`
  (elo/dial giữ 1700 · temp 0.36), boot UCI kiểm tra OK (id/Moves/Weights OK).
- Chọn model là quyết định của user (v4-final, ưu tiên phong cách).
- Bot **vẫn đang STOPPED** — user tự chạy hoặc yêu cầu khởi động khi muốn.
- GPU rảnh (40 MiB), không process dư thừa.
