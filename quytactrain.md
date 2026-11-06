# Hướng dẫn fine-tune Maia3-79M 



## 1.Quy trình tổng thể

1. Tải toàn bộ ván từ Chess.com.
2. Lọc ở cấp ván và gắn nhãn ở cấp nước đi.
3. Chia train / validation / test theo thời gian.
4. Đo baseline của mô hình gốc (nhiều mức elo).
5. Fine-tune, thử các biến thể (ablation).
6. Đánh giá trên test đúng một lần, kèm khoảng tin cậy.
7. Kiểm tra phong cách bằng cho bot tự chơi và kiểm tra mù.
8. Triển khai, rồi định kỳ thu thêm ván mới để huấn luyện lại.

## 2.Thu thập dữ liệu

- Chess.com cung cấp API công khai: `https://api.chess.com/pub/player/<ten>/games/archives` liệt kê các tháng, và `https://api.chess.com/pub/player/<ten>/games/YYYY/MM/pgn` trả về PGN của tháng đó. Nên gửi kèm header `User-Agent` có thông tin liên hệ của bạn.
- PGN của Chess.com có đủ các header cần dùng: `White`, `Black`, `WhiteElo`, `BlackElo`, `TimeControl`, `Link`, `UTCDate`, `Termination` và thời gian còn lại sau mỗi nước ở dạng chú thích `[%clk ...]`.

## 3.Lọc dữ liệu

### 3.1 Lọc cấp ván

| Tiêu chí | Gợi ý | Lý do |
| --- | --- | --- |
| Biến thể | Chỉ cờ tiêu chuẩn | Chess960, bughouse... có phân bố hoàn toàn khác |
| Nhịp độ | Rapid (từ 10 phút) giữ toàn bộ; blitz 3+0 đến 5+0 cân nhắc giữ với trọng số thấp; bullet bỏ | Nhịp càng nhanh, nước đi càng phản ánh phản xạ và lỗi thời gian hơn là phong cách |
| Rating | Từ khoảng 1250 trở lên (xem 3.3) | Phong cách thay đổi theo trình độ |
| Chênh lệch rating đối thủ | Giữ ván có chênh dưới 400 điểm | Đánh với đối thủ quá yếu hoặc quá mạnh làm lệch cách chơi |
| Độ dài | Ít nhất 10 ply | Loại ván bỏ cuộc, ván hủy |
| Kết thúc bất thường | Bỏ ván đối thủ rời bàn ngay đầu ván, ván bị hủy | Không có giá trị học |
| Trùng lặp | Khử theo header `Link` | Tránh một ván xuất hiện hai lần ở train và test |

### 3.2 Gắn nhãn và lọc cấp nước đi

Lưu mọi nước đi của bạn kèm siêu dữ liệu rồi lọc lúc nạp dữ liệu. Cách này cho phép thử nhiều ngưỡng mà không phải xử lý lại từ đầu.

- **Nước đi vội (premove hoặc dưới khoảng 1 giây)**: thường là nước bắt buộc hoặc phản xạ. Đề xuất bỏ hoặc giảm trọng số.
- **Cuối ván thiếu giờ**: khi còn dưới 10% thời gian gốc hoặc dưới 30 giây ở rapid. Nước đi lúc này không phản ánh phong cách. Đề xuất bỏ.
- **Nước bắt buộc**: thế cờ chỉ có một nước hợp lệ. Dễ đoán, làm tăng số đẹp một cách giả tạo. Bỏ khi huấn luyện, và cũng bỏ khỏi tập test chính.
- **Khai cuộc thuần thuộc lòng (6–8 ply đầu)**: giữ trong dữ liệu nhưng khi đánh giá phải báo riêng.
- **Thế đã quyết định (đánh giá trên ±800 centipawn)**: tùy chọn giảm trọng số, cần Stockfish để gắn nhãn.

### 3.3 Xử lý các ván cũ khi còn dưới 1200

Bỏ các ván này là hợp lý. Thay vì bỏ hẳn, hãy thử so sánh ba phương án bằng validation NLL trên các ván gần đây:

- **A**: bỏ hoàn toàn.
- **B**: giữ với trọng số thấp (khoảng 0,3–0,5) nếu code huấn luyện hỗ trợ trọng số mẫu, hoặc lấy mẫu ít hơn.
- **C**: huấn luyện hai giai đoạn: trước hết fine-tune trên toàn bộ ván, sau đó fine-tune tiếp trên ván gần đây với learning rate nhỏ hơn.



## 4. Chia tập dữ liệu

- **Chia theo ván và theo thời gian**: sắp xếp ván theo ngày, lấy khoảng 70–75% đầu làm train, khoảng 10% tiếp theo làm validation, khoảng 15% cuối làm test. Cách này mô phỏng đúng việc dùng thật: dự đoán ván tương lai từ ván quá khứ.
- **Không bao giờ chia theo nước đi**: các nước của cùng một ván nằm ở cả hai tập sẽ làm kết quả đẹp một cách sai lệch.
- **Tập test chỉ dùng một lần ở cuối**. Mọi lần chọn siêu tham số, chọn `--elo`, chọn epoch đều dựa trên validation.
- **Kiểm tra gần trùng**: so sánh 10–15 nước đầu của ván test với ván train. Nếu bạn hay lặp lại cùng một tuyến khai cuộc, số khai cuộc sẽ cao; đó là học thuộc lòng chứ chưa phải phong cách, vì vậy phải báo riêng (mục 9).
- **Lặp lại với nhiều cách chia**: cuộn cửa sổ thời gian 3–4 lần (rolling) để xem kết quả có ổn định không, vì chỉ một lần chia với vài trăm ván test có độ dao động đáng kể.

## 5. Chọn mô hình

Maia3 có các cỡ 5M, 23M, 79M (và bản ablation 3M). Với khoảng 60 nghìn nước đi, 79M không chắc là lựa chọn tốt nhất.

| Cỡ | Ưu điểm | Rủi ro |
| --- | --- | --- |
| 5M | Ít overfit, chạy nhanh trên CPU, dễ lặp thử nghiệm | Mô hình gốc kém chính xác hơn, có thể giới hạn mức tăng |
| 23M | Cân bằng | Cần early stopping cẩn thận |
| 79M | Mô hình gốc chính xác nhất | Dễ overfit với ít dữ liệu, tốn tài nguyên hơn |


## 6. Chiến lược huấn luyện

### 6.1 Trình tự đề xuất

1. **Baseline**: đo mô hình gốc trên validation ở nhiều `--elo` (mục 7). Ghi lại NLL, top-1, top-5.
2. **Fine-tune cơ bản**: toàn bộ tham số, learning rate nhỏ.
3. **Ablation**: lần lượt thử các biến thể ở 6.3, mỗi lần đổi một yếu tố.
4. **Nội suy trọng số** giữa mô hình gốc và bản fine-tune (6.4).
5. **Huấn luyện lại lần cuối** trên train + validation với số epoch đã chọn, đánh giá test một lần.

### 6.2 Siêu tham số khởi điểm (cần điều chỉnh theo kết quả)

| Tham số | Khởi điểm | Ghi chú |
| --- | --- | --- |
| Optimizer | AdamW |  |
| Learning rate | 1e-5 đến 5e-5 | Quét thử 3 giá trị; nhỏ hơn thì an toàn hơn |
| Lịch learning rate | Warmup 100–200 bước rồi giảm dần (cosine hoặc tuyến tính) |  |
| Batch size | 64–256 vị trí | Tùy bộ nhớ GPU, dùng tích lũy gradient nếu cần |
| Số epoch | 2–5 | Dừng theo validation NLL, không theo số epoch cố định |
| Weight decay | 0,01–0,1 |  |
| Gradient clipping | 1,0 |  |
| Độ chính xác | Mixed precision (bf16 hoặc fp16) |  |
| Seed | Chạy ít nhất 3 seed cho cấu hình cuối | Để biết dao động giữa các lần chạy |

Theo dõi validation sau mỗi nửa epoch hoặc mỗi epoch. Dừng khi validation NLL không giảm sau 1–2 lần đánh giá và lưu checkpoint tốt nhất.

### 6.3 Các biến thể nên thử

- **Điều kiện hóa rating**: dùng đúng rating ở từng ván (đã có trong dữ liệu) thay vì một giá trị cố định. Cần kiểm tra xem Maia3 dùng rating của người đi, của đối thủ hay cả hai, và truyền đúng như lúc huấn luyện mô hình gốc.
- **Đóng băng một phần**: giữ nguyên phần nhúng và vài khối đầu, chỉ cập nhật các khối cuối và đầu ra. Ít tham số cập nhật nên ít overfit hơn.
- **LoRA hoặc adapter**: nếu bạn tự thêm được. Số tham số huấn luyện rất nhỏ, thích hợp khi dữ liệu ít.
- **Trộn dữ liệu thay thế (replay)**: thêm 10–30% ván Lichess công khai của người chơi cùng mức rating (lấy từ `database.lichess.org`) để giảm hiện tượng quên kiến thức tổng quát. Tốn công chuẩn bị nhưng thường giúp ổn định.
- **Label smoothing nhẹ (0,0–0,1)**: giúp xác suất ít quá tự tin, hữu ích nếu sau này bạn lấy mẫu theo xác suất.
- **Trọng số mẫu**: giảm trọng số các nước dễ, nước vội, ván cũ; tăng cho thế cờ có nhiều nước đi hợp lý.
- **Tăng cường dữ liệu**: Maia3 có chuẩn hóa góc nhìn khi đến lượt Đen (bàn cờ được lật), nên đừng tự lật gương thêm nếu không hiểu rõ; lật trái phải làm hỏng luật nhập thành và phong cách.

### 6.4 Nội suy trọng số và gộp xác suất

Vì mô hình gốc và mô hình fine-tune cùng kiến trúc, bạn có thể tạo bản trộn: trọng số = (1 − α) × gốc + α × fine-tune, với α quét từ 0,5 đến 1,0. Thường ở mức giữa khoảng 0,7–0,9 cho validation NLL tốt hơn cả hai đầu mút, nhưng điều này không chắc chắn nên cần đo. Cách khác là trộn xác suất đầu ra của hai mô hình. Cả hai cách đều rẻ và giúp giảm overfit.

## 7. Vấn đề rating Chess.com và `--elo`

- Maia được huấn luyện trên ván Lichess. Với cùng một người, rating Lichess thường cao hơn Chess.com, mức chênh phụ thuộc từng người và nhịp độ, khoảng vài trăm điểm, và không có công thức chính xác.
- Vì vậy elo trong ván đấu có thể không tương ứng với elo đúng cần train. Khi fine-tune, mô hình có thể bù lại phần nào, nhưng vẫn nên quét.
- **Cách làm**: với mô hình gốc và với mô hình fine-tune, tính validation NLL và top-1 ở các mức `--elo` như 1450, 1550, 1650, 1750, 1850, 1950 (kiểm tra `maia3-uci --help` để biết khoảng giá trị được hỗ trợ). Chọn mức cho NLL thấp nhất.
- Nếu khi fine-tune bạn đã gán rating Chess.com vào đầu vào điều kiện, thì lúc chạy bot cũng phải dùng cùng thang đó. Nhất quán giữa huấn luyện và suy luận quan trọng hơn con số tuyệt đối.
- Khi rating dao động nhiều, có thể thử điều kiện theo rating trung bình của 30 ván gần nhất để giảm nhiễu.

## 8. Đánh giá hiệu quả

### 8.1 Chỉ số chính

| Chỉ số | Ý nghĩa | Ghi chú |
| --- | --- | --- |
| NLL (cross-entropy) | Trung bình −log xác suất mô hình gán cho nước thật | Nhạy nhất và đáng tin nhất; dùng làm tiêu chí chọn mô hình |
| Top-1 | Nước thật là nước mô hình ưu tiên nhất | Dễ hiểu nhưng bị thế dễ làm nhiễu |
| Top-3, Top-5 | Nước thật nằm trong 3 hoặc 5 nước đầu | Thường rất cao ở mọi mô hình, ít phân biệt được phong cách |
| Calibration (ECE) | Mô hình nói 60% thì thực tế đúng khoảng 60% không | Quan trọng nếu bot lấy mẫu theo xác suất |

### 8.2 Mốc so sánh bắt buộc

- Mô hình gốc ở mức `--elo` tốt nhất (không phải mức bạn đoán).
- Mô hình gốc ở nhiều elo, để thấy mức tăng thật.
- Stockfish (nước tốt nhất): tỷ lệ bạn đi trùng với Stockfish. Bot phong cách tốt phải khớp bạn hơn Stockfish rõ rệt.
- Nước phổ biến nhất trong dữ liệu của bạn theo từng thế (tra bảng), như mốc “học thuộc lòng”.

### 8.3 Chia nhỏ kết quả

- **Giai đoạn**: nước 1–10, 11–30, từ 31 trở đi. Nếu mức tăng chủ yếu ở nước 1–10 thì dùng opening book cho giai đoạn này và đừng tính nó vào đánh giá phong cách.
- **Độ khó thế cờ**: dùng entropy xác suất của mô hình gốc hoặc số nước hợp lệ để chia thế dễ và thế khó. Cải thiện ở thế khó mới phản ánh phong cách.
- **Màu quân**, **nhịp độ**, **chênh rating đối thủ**.
- **Loại nước đi**: bắt quân, chiếu, nhập thành, nước tốt, nước đi quân tốt.
- **Tiến trình theo thời gian**: so sánh ván cũ nhất và ván mới nhất trong test để xem phong cách của bạn có đổi không (nếu có, nên cập nhật mô hình định kỳ).

### 8.4 Độ tin cậy thống kê

Các nước trong cùng một ván không độc lập, nên phải bootstrap **theo ván**: lấy mẫu có hoàn lại các ván, tính lại chỉ số, lặp 2000 lần, lấy khoảng 2,5%–97,5%. Khi so hai mô hình, bootstrap độ chênh trên **cùng** các ván đã lấy mẫu (paired bootstrap).

```python
import numpy as np, chess

def evaluate(samples, predict):
    # predict(board, elo) -> dict {uci: prob} trên các nước hợp lệ, tổng bằng 1
    # hàm này bạn tự nối với code suy luận Maia3 của mình
    rows = []
    for s in samples:
        board = chess.Board(s['fen'])
        probs = predict(board, s['elo'])
        ranked = sorted(probs, key=probs.get, reverse=True)
        p = max(probs.get(s['move'], 0.0), 1e-9)
        rows.append({
            'game': s['game'], 'ply': s['ply'],
            'top1': ranked[0] == s['move'],
            'top3': s['move'] in ranked[:3],
            'top5': s['move'] in ranked[:5],
            'nll': -float(np.log(p)),
            'n_legal': len(probs),
        })
    return rows

def per_game(rows, key):
    d = {}
    for r in rows:
        a = d.setdefault(r['game'], [0.0, 0])
        a[0] += float(r[key])
        a[1] += 1
    games = sorted(d)
    return games, np.array([d[g][0] for g in games]), np.array([d[g][1] for g in games])

def bootstrap_diff(rows_a, rows_b, key, n=2000, seed=0):
    # rows_a và rows_b phải cùng tập mẫu, cùng thứ tự
    games, sa, ca = per_game(rows_a, key)
    _, sb, cb = per_game(rows_b, key)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(games), len(games))
        diffs.append(sa[idx].sum() / ca[idx].sum() - sb[idx].sum() / cb[idx].sum())
    return float(np.mean(diffs)), np.percentile(diffs, [2.5, 97.5])

def by_phase(rows):
    out = {}
    for name, lo, hi in [('khai cuộc', 0, 20), ('trung cuộc', 20, 60), ('tàn cuộc', 60, 10**6)]:
        sub = [r for r in rows if lo <= r['ply'] < hi]
        if sub:
            out[name] = {'n': len(sub),
                         'top1': float(np.mean([r['top1'] for r in sub])),
                         'nll': float(np.mean([r['nll'] for r in sub]))}
    return out
```

`ply` ở đây tính theo nước nửa (ply), nên 20 ply tương đương 10 nước đầy đủ. Hàm `predict` phụ thuộc vào API suy luận trong repo Maia3; nếu chỉ lấy được nước đi từ engine UCI, bạn vẫn tính được top-1 nhưng không tính được NLL.

### 8.5 Đánh giá phong cách khi bot tự chơi

Chỉ số dự đoán nước đi chưa đủ. Hãy cho bot chơi khoảng 200–300 ván (bot đấu bot với lấy mẫu theo xác suất, hoặc đấu Stockfish giới hạn sức mạnh) rồi so các phân bố sau với ván thật của bạn:

- Phân bố nước đầu và khai cuộc theo từng màu.
- Độ dài ván trung bình, thời điểm đổi Hậu, thời điểm nhập thành.
- Số quân bị bắt và tỷ lệ đổi quân.
- Mất điểm trung bình (average centipawn loss) và tỷ lệ sai lầm nặng (mất trên 200 centipawn) tính bằng Stockfish với độ sâu cố định.
- Tỷ lệ thắng, hòa, thua so với mức Stockfish tương đương rating của bạn.

Với các phân bố rời rạc, tính khoảng cách Jensen–Shannon giữa bot và bạn, và so với khoảng cách giữa mô hình gốc và bạn. Bot tốt khi khoảng cách nhỏ hơn rõ rệt.

### 8.6 Kiểm tra mù

| Tình huống | Chẩn đoán | Việc nên làm |
| --- | --- | --- |
| NLL giảm, top-1 tăng ở cả ba giai đoạn | Học được phong cách thật | Chốt cấu hình |
| Chỉ khai cuộc tăng | Mới học thuộc repertoire | Dùng opening book cho khai cuộc |
| Train tốt, validation xấu dần | Overfit | Giảm learning rate, giảm epoch, đóng băng một phần, thêm replay, thử cỡ nhỏ hơn |
| Top-1 tăng nhưng NLL tăng | Mô hình quá tự tin | Label smoothing, nội suy trọng số |
| Tốt trên test nhưng bot chơi không giống | Chỉ số dự đoán không nắm bắt hết phong cách | Xem lại 8.5, thử lấy mẫu theo xác suất |

## 9. Triển khai và kiểm tra thực tế

- Trong tập lệnh chạy engine, đặt `--model maia3-79m` , `--checkpoint-path` trỏ đúng tới checkpoint, `--use-uci-history`, và `--elo` bằng mức đã chọn ở mục 7.
- Nếu bot đi y hệt mỗi ván và bạn muốn đa dạng, cân nhắc lấy mẫu theo xác suất ở những thế có nhiều nước hợp lý, và luôn chọn nước tốt nhất ở thế có một nước vượt trội. Cần kiểm tra xem `maia3-uci` hỗ trợ tùy chọn lấy mẫu nào (xem `--help`), vì mình chưa xác minh.

## 10. Kế hoạch thử nghiệm gợi ý

Ghi mọi thí nghiệm vào một bảng để so sánh, đổi một yếu tố mỗi lần.

| # | Thí nghiệm | Cỡ | Dữ liệu | Học | Kết quả cần ghi |
| --- | --- | --- | --- | --- | --- |
| 0 | Mô hình gốc, quét elo | 5M, 23M, 79M | validation | không | NLL, top-1 ở từng elo |
| 1 | Fine-tune cơ bản | 79M | ván ≥ 1250 | lr 2e-5 | NLL, top-1 theo giai đoạn |
| 2 | Như 1, cỡ khác | 23M, 5M | như 1 | như 1 | so với 1 |
| 3 | Learning rate | cỡ tốt nhất | như 1 | 1e-5, 5e-5 | NLL validation |
| 4 | Đóng băng một phần | cỡ tốt nhất | như 1 | như 3 | NLL validation |
| 5 | Thêm ván cũ có trọng số thấp | cỡ tốt nhất | A, B, C (mục 4.3) | như 3 | NLL trên ván gần đây |
| 6 | Replay Lichess | cỡ tốt nhất | thêm 10–30% | như 3 | NLL, top-1 |
| 7 | Nội suy trọng số | cỡ tốt nhất | như tốt nhất | α 0,5–1,0 | NLL |
| 8 | Huấn luyện lại train + validation | cấu hình cuối | tất cả | cấu hình cuối | test một lần, kèm khoảng tin cậy |

## 11. Lỗi thường gặp

- **Rò rỉ dữ liệu** do chia theo nước đi, hoặc cùng ván nằm ở cả train và test (đặc biệt khi tải từ nhiều nguồn).
- **So với mốc yếu**: chỉ so với mô hình gốc ở một elo bất kỳ.
- **Chọn siêu tham số trên test** rồi báo số trên chính test đó.
- **Quá tin vào top-5**: gần như mô hình nào cũng đạt mức rất cao, nên không phân biệt được phong cách.
- **Không xử lý nước vội và cuối ván thiếu giờ**: làm bot học các “lỗi thời gian” chứ không phải phong cách.
- **Thang rating không nhất quán** giữa huấn luyện và suy luận.
- **Chạy một seed duy nhất** rồi kết luận từ chênh lệch nhỏ hơn dao động giữa các lần chạy.
- **Nạp checkpoint sai cỡ** hoặc sai cấu hình kiến trúc: lỗi kích thước tensor.
- **Bot quá mạnh**: kiểm tra có vô tình bật tính nhiều node hoặc đang dùng engine khác không; Maia3 dùng một lần suy luận mỗi nước nên không nên mạnh hơn bạn đáng kể.

## 12. Danh sách kiểm tra cuối

- [ ] Dữ liệu tải đầy đủ, khử trùng theo `Link`, chỉ cờ tiêu chuẩn.
- [ ] Lọc cấp ván và gắn nhãn cấp nước đi bằng script, lưu lại tham số lọc.
- [ ] Chia theo ván và theo thời gian; test chưa từng được dùng để chọn gì.
- [ ] Đã đo baseline mô hình gốc ở nhiều `--elo` và nhiều cỡ.
- [ ] Đã thử ít nhất: 2 learning rate, 1 cấu hình đóng băng, nội suy trọng số.
- [ ] Đã chạy 3 seed cho cấu hình cuối và có khoảng tin cậy bootstrap theo ván.
- [ ] Đã báo kết quả riêng cho khai cuộc, trung cuộc, tàn cuộc và thế khó.
- [ ] Đã so sánh phân bố phong cách của bot tự chơi với ván thật.
- [ ] Đã chọn `--elo` bằng validation, không bằng cảm tính.
- [ ] Đã chơi thử ván thật và (nếu có thể) kiểm tra mù.
- [ ] Có kế hoạch huấn luyện lại định kỳ khi có thêm khoảng 500 ván mới.
