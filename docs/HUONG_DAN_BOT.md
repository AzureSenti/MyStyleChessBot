# 📘 HƯỚNG DẪN VẬN HÀNH & TINH CHỈNH BOT CỜ VUA PHONG CÁCH CÁ NHÂN

> Bot của bạn = **Maia3-79M pretrained** ✓ được fine-tune bằng **1.879 ván rapid**
> (TsukinoLuna + Sasayaku, 2022-09 → 2026-10) trên RTX 4060 Laptop 8GB.

---

## 1. Bản đồ hệ thống

| Thành phần | File / thư mục | Vai trò |
|---|---|---|
| Tải ván cờ Chess.com | `code/download_chesscom_games.py` | PGN theo tháng + file gộp, lọc `--tc/--result/--color` |
| Chuyển PGN → tensor | `code/prepare_dataset.py` | Dataset chuẩn đầu vào Maia3 (`tokens/legal_mask/elo/value`) |
| Fine-tune | `code/finetune.py` | Xuất checkpoint `best.pt/last.pt/epochN.pt` |
| Engine UCI | `.venv/bin/maia3-uci` (từ `maia3/`) | Chơi với GUI/web/lichess-bot, có Elo·Temperature·TopP·MultiPV |
| Web UI chơi thử | `code/playui.py` | http://localhost:8765 — click-click trên bàn cờ |
| Bot Lichess | `.lichess-bot/` + `.venv-bot/` + `code/config.yml` | AzuPre tự nhận challenge trên lichess.org |
| Engine wrapper cho bot | `code/maia3-bot-engine.sh` | Ghim sẵn checkpoint + elo + temperature |
| Dataset | `data/datasets/rapid_style/` | 129.924 positions (train 123k / val 6.8k) |
| Checkpoint hiện tại | `finetunes/maia3-79m/best.pt` | val top1 60,09% (baseline 58,64%), value acc 80% |

Môi trường Python: có **2 venv riêng**: `.venv` (torch + maia3) và `.venv-bot` (lichess-bot).

---

## 2. Khởi động nhanh

```bash
cd /mnt/General_Data/Projects/Python/maia3

# (A) Chơi cục bộ qua web (nhanh, để kiểm tra tương tác)
.venv/bin/python code/playui.py --checkpoint finetunes/maia3-79m/best.pt
#    → mở http://localhost:8765 trên máy hay điện thoại cùng WiFi (xem IP: hostname -I)

# (B) Chạy bot Lichess (AzuPre nhận mọi challenge tự động)
.venv-bot/bin/python .lichess-bot/lichess-bot.py --config code/config.yml

# Dừng bất kỳ tiến trình nào:
pkill -9 -f "li[c]hess-bot" ; pkill -9 -f "[p]layui.py" ; pkill -9 -f "[m]aia3-uci"
```

> ⚠️ **Quy tắc sống còn của Lichess: chỉ được chạy 1 process bot mỗi tài khoản.**
> Chạy ≥2 process (hoặc còn sót stream cũ) → Lichess trả
> `{"error": "Please don't poll this endpoint..."}` → bot nhận challenge nhưng treo, không đi nước nào.
> Khắc phục: kill hết → **chờ ~2 phút** (cooldown) → chạy lại đúng 1 process.

---

## 3. Quy trình chuẩn: dữ liệu → huấn luyện → xuất bot

### 3.1 Tải / cập nhật dữ liệu ván của bạn
```bash
# Rapid, cả thắng/hòa/thua, từ 2023-01 → nay:
.venv/bin/python code/download_chesscom_games.py TsukinoLuna \
    --out data/chesscom --tc rapid --from 2023-01 --force
# Thêm tài khoản phụ:
.venv/bin/python code/download_chesscom_games.py Sasayaku --out data/chesscom --tc rapid

# Gộp thành dataset:
cat data/chesscom/TsukinoLuna_ALL.pgn data/chesscom/Sasayaku_ALL.pgn \
    > data/chesscom/rapid_dataset_ALL.pgn
```
Muốn lọc theo kết quả (ví dụ chỉ lấy thắng + hòa): thêm `--result win --result draw`
(có thể lặp `--result` nhiều lần). Script có cache tháng — `--force` để tải lại từ đầu.

**Lọc ván theo chất lượng (Stockfish)** — loại ván tilt / ẩu trước khi train:
```bash
# ⚠️ tải blitz PHẢI dùng thư mục riêng + --force (cache tháng không phân biệt tốc độ):
.venv/bin/python code/download_chesscom_games.py TsukinoLuna \
    --out data/chesscom/blitz --tc blitz --from 2023-01 --force

# gộp rapid + blitz rồi lọc (đánh giá riêng các nước BẠN đi):
cat data/chesscom/rapid_dataset_ALL.pgn data/chesscom/blitz/*_ALL.pgn \
    > data/chesscom/all_rapid_blitz.pgn
.venv/bin/python code/filter_games.py --pgn data/chesscom/all_rapid_blitz.pgn \
    --accounts TsukinoLuna Sasayaku \
    --out-pgn data/chesscom/all_filtered.pgn \
    --out-csv data/chesscom/game_quality_all.csv --workers 12
```
Gate mặc định: accuracy (thang win% của script) — **THẮNG/HÒA ≥ 80, THUA ≥ 84**
(≈ thang Chess.com hiển thị 55/60, vì thang script cao hơn Chess.com ~25 điểm).
Muốn chặt hơn: `--min-accuracy-win 85 --min-accuracy-loss 90 --max-avg-cpl 120 --max-blunder-rate 0.15`.
CSV có cột `outcome` + `accuracy` từng ván của riêng bạn — mở để tự chỉnh ngưỡng.
Sau khi lọc, chỉ `prepare_dataset.py` trên `all_filtered.pgn` thay cho PGN gộp thô.

### 3.2 Tạo dataset tensor
```bash
.venv/bin/python code/prepare_dataset.py data/chesscom/rapid_dataset_ALL.pgn \
    --accounts TsukinoLuna Sasayaku --out data/datasets/rapid_style
# Kiểm tra nhanh bằng meta.json: tổng positions, số train/val.
```
Định dạng đầu ra: tokens `(N, 64, 97)` fp16 + legal_mask (packbits) + label `move_idx` +
`value_idx` + `elo` + `game_id`; **split train/val theo game** (không chia lẻ position)
để val phản ánh trung thực. Tỉ lệ chia chỉnh được bằng `--val-fraction`.

### 3.3 Fine-tune (cấu hình đã tối cho RTX 4060 8GB)
```bash
# Setup đã kiểm chứng trên máy này:
.venv/bin/python code/finetune.py --dataset data/datasets/rapid_style \
    --model maia3-79m --batch-size 128 --no-checkpointing --workers 4 \
    --epochs 4 --lr 1e-5 --value-weight 0.3
# Tốc độ: ~390 pos/s → 1 epoch 79M ≈ 5.5 phút; eval Top-1 mỗi 240 bước.
```

| Model | 1 epoch | 4 epochs | VRAM (đo thực) |
|---|---|---|---|
| maia3-5m  | ~1 phút  | ~4 phút  | ~0,4 GB |
| maia3-23m | ~3 phút  | ~12 phút | ~0,5 GB |
| maia3-79m | ~5.5 phút | ~21 phút | ~2,5–4 GB |

**Chống overfit (đã xảy ra 1 lần: best nằm ở cuối epoch 0):**
- `--freeze-embeddings`: giữ nguyên embedding (Elo + token) — giảm số tham số được train.
- `--train-blocks-from 6`: chỉ train 2 block cuối (tầng "phong cách" cao) → ít overfit hơn.
- LR an toàn: `5e-6` (nhẹ) → `2e-5` (nóng, dễ overfit). LR thấp + ít epoch là chuẩn cho lần retrain sau.
- Luôn so sánh val top1 / policy CE giữa các epoch — chỉ tin checkpoint tốt nhất trên **val**.
- `--resume` tiếp tục từ `last.pt` (kèm optimizer state) nếu bị dừng giữa chừng.

### 3.4 Chọn checkpoint để bot sử dụng
```bash
# Trong finetunes/maia3-79m/:
best.pt     ← top1 val cao nhất      → dùng cho bot (khuyên dùng, ~318 MB)
last.pt     ← model + optimizer (~912 MB) → chỉ dùng cho --resume, không dùng để chơi
epoch{N}.pt ← lưu để so sánh giữa các epoch
```

---

## 4. Tinh chỉnh tính cách bot theo ý bạn

### 4.1 Qua tham số engine (nhanh nhất — sửa `code/maia3-bot-engine.sh` rồi restart bot)
| Option | Ý nghĩa | Khuyến nghị |
|---|---|---|
| `--elo` | Elo "tự cảm" của model (SelfElo = OppoElo) | Elo thực của bạn (**hiện đang set 1700**) — đặt về 1460 nếu muốn bot yếu đúng trình bạn |
| `--temperature` | 0 = luôn chọn nước xác suất cao nhất · cao = đa dạng nhưng dễ đi "vớ" | **hiện 0.36** — chuẩn; 0.5–0.8 nếu muốn đa dạng hơn |
| `--top_p` | Nucleus sampling: 1.0 = tắt; <1 cắt đuôi phân bố | 0.95 nếu temperature > 0.5 |
| `--multipv` | Số biến phương án in ra `info` (cho UI/GUI) | 5 |

Ví dụ bot "giống bạn nhất" (luôn chọn gần nước bạn thật sự hay đi):
```bash
maia3-uci --model maia3-79m --checkpoint-path finetunes/maia3-79m/best.pt \
  --use-uci-history --elo 1460 --temperature 0.2
```
Bot " wildcard " (đa dạng, khó đoán hơn chính bạn):
```bash
... --temperature 0.8 --top-p 0.9
```

### 4.2 Qua `code/config.yml` (bot Lichess)
Sửa file → restart bot để áp dụng:

| Tham số | Hiện tại | Ý nghĩa |
|---|---|---|
| `engine.dir` / `engine.name` | `code/` / `maia3-bot-engine.sh` | Muốn đổi checkpoint/Elo/Temperature → sửa nội dung wrapper script |
| `challenge.min_increment` / `max_increment` | 0 / 180 | Giới hạn increment của challenge được nhận |
| `challenge.min_base` / `max_base` | 0 / lớn | Giới hạn thời gian cơ bản (giây) — ví dụ đặt min 180 để chỉ nhận rapid+ |
| `challenge.time_controls` | bullet, blitz, rapid | Nhịp chấp nhận (bỏ bullet nếu muốn) |
| `challenge.modes` | casual, rated | Casual hoặc rated hoặc cả hai |
| `challenge.accept_bot` | true | Nhận challenge từ bot khác |
| `challenge.concurrency` | 1 | Số ván song song (giữ 1 cho ổn định stream) |
| `challenge.variants` | standard | Maia chỉ chơi standard — đừng thêm variant khác |
| `fake_think_time` | false | true = bot "suy nghĩ" giả cho tự nhiên như người |
| `move_overhead` | 2000 ms | Tăng nếu mạng chậm hay bị tính forfeit |
| `greeting.hello` / `goodbye` | (tiếng Việt) | Lời chat đầu / cuối ván |

Cho bot **tự mời đối thủ** (không cần ai challenge): thêm mục `matchmaking`:
```yaml
matchmaking:
  search_timeout: 120    # giây chờ giữa các lần tìm đối thủ
  minimum_wait: 60       # giây chờ tối thiểu sau ván trước
  allow_bot: false       # true nếu muốn bot đi đấu bot khác
```
(Tham khảo đầy đủ tại wiki repo: https://github.com/lichess-bot/lichess-bot/wiki)

### 4.3 Đổi "linh hồn": thử checkpoint khác
- Web UI với epoch khác: `.venv/bin/python code/playui.py --checkpoint finetunes/maia3-79m/epoch1.pt`
- So với **baseline gốc** (chưa fine-tune): bỏ `--checkpoint`, chỉ giữ `--model maia3-79m --elo 1460`.

### 4.4 Đo "bot giống mình đến đâu" (khách quan)
Xem `finetunes/maia3-79m/stats.jsonl` sau mỗi lần train:
- `top1`: % lúc model đoán đúng nước bạn đã đi — mục tiêu > 0.60 (baseline 0.586, hiện 0.6009).
- `policy_ce`: thấp hơn = phân phối nước đi của model gần bạn hơn (chỉ số nhạy nhất).
- `value_acc`: khả năng đọc vị thắng/thua theo diễn biến ván.

*Lưu ý thực tế: con người cùng thế cờ cũng có 3–4 nước "bình thường" khác nhau, và ~60% top-1
đã là mức rất "human" — đừng kỳ vọng 90% như engine cứng.*

---

## 5. Vòng lặp "cho bot học thêm về tôi" (khi có ván mới)

```bash
# 1. Tải ván mới (chỉ tháng mới được tải nếu không --force toàn bộ)
.venv/bin/python code/download_chesscom_games.py TsukinoLuna --out data/chesscom --tc rapid
# 2. Gộp lại + rebuild dataset
cat data/chesscom/TsukinoLuna_ALL.pgn data/chesscom/Sasayaku_ALL.pgn \
    > data/chesscom/rapid_dataset_ALL.pgn
.venv/bin/python code/prepare_dataset.py data/chesscom/rapid_dataset_ALL.pgn \
    --accounts TsukinoLuna Sasayaku --out data/datasets/rapid_style
# 3. Fine-tune lại từ checkpoint gốc (nên, tránh học đè overfit) HOẶC --resume từ last.pt
.venv/bin/python code/finetune.py --dataset data/datasets/rapid_style --model maia3-79m \
    --batch-size 128 --no-checkpointing --workers 4 --epochs 4 --lr 1e-5 --value-weight 0.3
# 4. Restart bot Lichess / playui để nạp best.pt mới
```
Khuyến nghị: sau ~20–30 ván mới trên Lichess thì chạy lại vòng này một lần.

---

## 5b. Kiểm thử đa đối thủ (quy tắc §8.5/§9)

### (A) Trên Lichess — thực chiến, tự thu PGN

```bash
# 1. Bot PHẢI đang chạy (.venv-bot/bin/python .lichess-bot/lichess-bot.py — mục 7)

# 2. Mời đối thủ định sẵn (đổi màu mỗi ván, chờ từng ván xong):
.venv/bin/python code/lichess_challenges.py challenge \
    --opponents maia1 maia5 LunaBaka --games-per-opp 2 --tc 300+2
#   thêm --rated nếu muốn tính rating; maia1/maia5 = 2 bot Maia "chính chủ"
#   (rapid ~1528/1635) — đối thủ cùng họ, đa phong cách

# 3. Tải PGN tất cả ván bot (nối vào file): dùng lại cho replay vòng 2 (§9)
.venv/bin/python code/lichess_challenges.py import --since 2026-11-05 \
    --out data/chesscom/azu_pre/lichess.pgn
```

Lưu ý A: bot chỉ chơi 1 ván/lúc (config `challenge.concurrency: 1`) — script
tôn trọng điều này; mời người thường → chỉ CASUAL (không --rated); respect
rate limit (script chờ mặc định giữa 2 mời).

### (B1) SF ladder — đường cong sức chịu đòn cục bộ

```bash
.venv/bin/python code/sf_ladder.py \
    --checkpoint finetunes/maia3-79m-v4-final/best.pt \
    --elos 1200 1400 1600 1800 2000 2200 --games-per-elo 16 \
    --out data/chesscom/ladder_v4final.pgn --json docs/sf_ladder_v4final.json
```

- 16 ván/mức (8 Trắng/8 Đen), khai seed từ corpus, temp/dial = cấu hình deploy
- Ra bảng điểm-% mỗi elo + **Elo ước lượng** (nội suy điểm 50%)
- GPU-safety: `--concurrency 2` → tối đa 2 maia engine (đừng tăng vượt 2 nếu
  có bot/web UI đang chạy cùng)

---

## 6. Xử lý sự cố
So sánh giữa model: chạy lại lệnh với `--checkpoint` khác, đổi tên `--out/--json`.


| Triệu chứng | Nguyên nhân | Giải pháp |
|---|---|---|
| Bot nhận challenge nhưng **không đi nước nào**, log lặp `Please don't poll this endpoint` | Đang chạy >1 process bot, hoặc stream cũ chưa chết; Lichess chỉ cho 1 stream/tài khoản | `pkill -9 -f "li[c]hess-bot"`, `pkill -9 -f "[m]aia3-uci"` → chờ 2 phút → chạy lại 1 process |
| Ván cũ treo mãi, muốn dọn | Ván rác do lỗi stream trước đây | `curl -X POST -H "Authorization: Bearer $TOKEN" https://lichess.org/api/bot/game/<gameId>/resign` |
| GUI desktop (En Croissant,…) load engine lâu / timeout | Model 318 MB load chậm lần đầu | Đợi lần đầu (sandbox), hoặc mở qua Web UI trước để "nóng" model |
| `engine does not support option Threads` (python-chess / lichess-bot) | Engine cũ chưa khai báo đủ option UCI | **ĐÃ FIX** — `maia3/uci.py` giờ khai báo Threads/Hash/Move Overhead/UCI_Opponent; engine ở `.venv/bin/maia3-uci` cần reinstall nếu gặp lại |
| Engine trả `bestmove 0000` hoặc illegal move | Move history không khớp / board sai trạng thái | Đảm bảo `--use-uci-history` (lichess-bot auto), kiểm tra position gửi vào engine |
| Bot in "Maia3 ready" nhưng không trả bestmove | Checkpoint path sai hoặc GPU bận | Xem stderr: `loading Maia3 checkpoint …`; xác nhận `best.pt` tồn tại |
| Web UI không mở được (port 8765 bận) | Process playui cũ còn sống | `pkill -9 -f "[p]layui.py"` rồi chạy lại |

---

## 7. Cheat-sheet (copy chạy)

```bash
cd /mnt/General_Data/Projects/Python/maia3

# 🟢 CHẠY BOT LICHESS (1 process duy nhất!)
.venv-bot/bin/python .lichess-bot/lichess-bot.py --config code/config.yml

# 🟢 WEBSITE CHƠI CỤC BỘ
.venv/bin/python code/playui.py --checkpoint finetunes/maia3-79m/best.pt

# 🟡 KIỂM TRA ENGINE NHANH (phải ra "bestmove <nước>"):
echo -e "uci\nisready\nposition startpos moves e2e4\ngo movetime 1000\nquit" \
  | ./code/maia3-bot-engine.sh 2>/dev/null | tail -3

# 🟡 ĐỔI TÍNH CÁCH BOT (SelfElo / temperature): sửa file này rồi restart bot
nano code/maia3-bot-engine.sh

# 🔴 DỪNG TẤT CẢ
pkill -9 -f "li[c]hess-bot"; pkill -9 -f "[p]layui.py"; pkill -9 -f "[m]aia3-uci"
```

---

*Tài liệu tạo 2026-10-30 — dataset 129.924 positions · fine-tune 21,3 phút trên RTX 4060 ·
val top1 60,09% (baseline 58,64%) · value-acc 80% · engine UCI tương thích lichess-bot 1.2.0.*
