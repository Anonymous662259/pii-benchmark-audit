# PROTOCOL — PII rare-type experiments (bản nháp, chờ duyệt)

> Trạng thái: **DRAFT 2026-09-27**. Chưa có hiệu lực cho tới khi được duyệt và commit.
> Sau khi commit, mọi thay đổi phải ghi vào mục 10 (nhật ký sai lệch) kèm ngày và lý do.
> Giai đoạn 1 chỉ được bắt đầu SAU commit đó.

## 0. Cần bạn quyết định trước khi commit

| # | Quyết định | Đề xuất |
|---|---|---|
| D1 | Hai seed đã chạy ở v1 (42, 1337) có dùng lại không? Chúng không có dump threshold thấp và checkpoint đã bị xoá. | **Chạy lại** cả 5 seed cho mọi arm với dump. Run v1 được coi là pilot và nói rõ như vậy trong paper. |
| D2 | Có chạy arm (d) "3 term, không có thành phần phụ" không? | Không ở Giai đoạn 1. Chỉ chạy nếu kết cục là A và cần tách tiếp. |
| D3 | Ngân sách GPU và cách chia workspace (mục 8) | Theo bảng ở mục 8 |

## 1. Câu hỏi

Trên 4 corpus PII (17–42 loại), các loại hiếm có F1 thấp hơn trung bình (BCE v1 pilot:
macro 90.68, rare-half 86.12). **Thành phần nào của training làm tăng F1 cho loại hiếm,
và phải trả giá bằng gì (micro-F1, số lượng miss ở loại phổ biến)?**

## 2. Các arm (GLiNER2, `fastino/gliner2-multi-v1`)

| Arm | Recipe (`scripts/modal_pii_baselines.py` RECIPES) | Cờ train |
|---|---|---|
| (a) | `bce` | *(không có)* |
| (b) | `bce_bundle` | `--loss bce --loss-class-weight --pos-weight 2 --boundary-smoothing 0.1 --boundary-smoothing-size 1 --null-anchor --full-grid` |
| (c) | `set` | `--loss set --set-topk 0 --loss-class-weight --pos-weight 2 --boundary-smoothing 0.1 --boundary-smoothing-size 1` (null anchor và full grid tự bật) |

(c) − (b) chỉ khác đúng ba term span / type / fp. (b) − (a) là phần của các thành phần phụ
**cộng với việc tắt negative masking**: BCE gốc của GLiNER2 che ngẫu nhiên 50% ô âm mỗi bước
(`masking_rate=0.5`), còn `set` luôn tính trên toàn lưới. `--full-grid` (thêm 2026-09-27) đưa
arm (b) về cùng lưới với (c).
Siêu tham số của SCL giữ mặc định trong code: λ_span = 1.0, λ_type = λ_fp = 0.25,
m_span = m_type = 0.5, τ = 0.5. Không tune thêm trong vòng này.

## 3. Dữ liệu (không đổi so với pilot)

| Key | Repo | Nhãn | train / val / test |
|---|---|---|---|
| `privy` | `anonymous-org/pii-privy` | 17 | 50000 / 2500 / 5000 |
| `pii_500k` (OpenPII) | `anonymous-org/pii-500k` | 20 | 49998 / 2500 / 4999 |
| `kiji` | `anonymous-org/pii-kiji` | 28 | 43845 / 2500 / 5000 |
| `gretel` | `anonymous-org/pii-gretel` | 42 | 50000 / 2500 / 5000 |

Tiền xử lý do `scripts/prepare_pii_datasets.py` thực hiện (cap độ dài train 2000 ký tự, bỏ nhãn
có dưới 20 mẫu train). Không sửa dữ liệu giữa chừng.

## 4. Training (giữ giống nhau cho cả 3 arm)

- 3 epoch, effective batch 16, lr encoder 1e-5 / task 5e-4, schedule linear với warmup 0.1
  (giá trị `DEFAULT_TRAINING`).
- GPU, micro-batch và dtype theo dataset, **cố định** trong `modal_pii_baselines.py`:
  A10G, 8×2, fp16 là mặc định; privy dùng A100-80GB, 4×4, bf16; gretel dùng A100-40GB, 4×4.
- **Seed: 42, 1337, 2024, 7, 13.** Báo cáo cả 5, không chọn, không bỏ seed nào.
  Nếu một run hỏng (NaN, OOM), chạy lại *đúng seed đó*, ghi vào mục 10.
- Checkpoint của seed 42 được giữ trên Volume (`--keep-ckpt-seeds 42`).

## 5. Decode và chấm điểm

- Decode chính thức: `flat_ner`, greedy, threshold **p0 = 0.5** cho cả 3 arm.
- Mọi split còn được decode thêm ở **threshold 0.1** (`--dump-threshold 0.1`, thư mục
  `<split>_t0.1/`) để phân tích threshold offline.
- Scorer: `HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)` (strict exact match).
- **Rare-half**: nửa số nhãn có support thấp nhất *trên test* của từng corpus (giống pilot).
  Kiểm tra độ nhạy bằng cách chia theo tần suất train.

## 6. Endpoint

- **Chính:** rare-half macro-F1, lấy trung bình 4 corpus, cho phép so **(c) − (b)** ở
  **threshold chọn trên validation** (mục 6.1).
- **Phụ:** macro-F1, micro-F1, số lỗi thay đổi theo nhóm support
  (`scripts/error_breakdown.py`), và cả ba phép so (b)−(a), (c)−(a), (c)−(b), ở cả hai
  chế độ threshold.

### 6.1 Luật chọn threshold

Với mỗi (arm, corpus, seed): chọn một threshold toàn cục t\* trong lưới {0.10, 0.15, …, 0.90}
sao cho **macro-F1 trên validation** cao nhất. Sau đó báo số trên test ở t\*.

Điều kiện tiên quyết: lọc dump 0.1 theo t phải cho kết quả **trùng** với decode lại ở t.
Dry test kiểm tra điều này (mục 9). Nếu không trùng, phải decode lại ở từng t trong lưới,
và ghi vào mục 10.

## 7. Thống kê và luật quyết định

- Paired bootstrap trên câu test, 5000 lần resample, mỗi lần trung bình theo seed của từng
  arm, dùng chung cách lấy mẫu cho cả hai arm (`scripts/pii_significance.py`). Báo CI 95% và p.
- **Kết cục A** ("3 term giúp thêm"): endpoint chính > 0 **và** cận dưới CI 95% > 0.
- **Kết cục B** (ngược lại): phần tăng đến từ các thành phần phụ; paper báo cáo 3 term là
  một thành phần đã được kiểm tra.
- Dù ra kết cục nào, **vẫn báo cáo đầy đủ**: cả 3 arm, cả 5 seed, cả hai chế độ threshold,
  kể cả khi micro-F1 giảm và khi kết quả khác nhau giữa các corpus.

## 8. Ngân sách (ước từ `elapsed_seconds` của pilot)

BCE khoảng 1.4 giờ/run, SCL khoảng 1.9 giờ/run; `bce_bundle` giả định khoảng 1.6 giờ.
Dump thêm ở 0.1 làm thời gian eval tăng lên, chưa đo được (sẽ có sau dry test).

| Trường hợp | Số run | GPU-giờ (ước) |
|---|---|---|
| D1 = chạy lại cả 5 seed | 3 × 4 × 5 = 60 | khoảng 100 |
| D1 = dùng lại 2 seed pilot | 44 | khoảng 70 |

Hạn mức $30 tính **theo workspace**. Kiểm tra cột workspace của `modal profile list` trước
khi giao việc. Dry test chạy trên profile `<workspace>` (workspace `<workspace>`).

## 9. Dry test (2026-09-27, `<workspace>`)

`launch_pii_jobs.py --smoke --datasets pii_500k --families gliner2 --seeds 42
--dump-threshold 0.1`, chạy lần lượt cho `bce`, `bce_bundle`, `set` (64 mẫu train, 32 mẫu eval).

Kiểm tra:
1. Cả 3 recipe chạy xong và có `record.json` với `status=ok`.
2. Log train cho thấy đúng loss và thành phần được bật (class weight, null anchor).
3. Có đủ `validation/`, `test/`, `validation_t0.1/`, `test_t0.1/`.
4. Lọc `test_t0.1` ở 0.5 trùng với `test/` (điều kiện của mục 6.1).

Kết quả (dữ liệu ở `out_pii_baselines/smoke_v2/`):

| Kiểm tra | bce | bce_bundle | set |
|---|---|---|---|
| 1. `status=ok`, thời gian | ✅ 1.8 phút | ✅ 2.0 phút (chạy lại với `--full-grid`) | ✅ 2.0 phút |
| 2. Log loss | BCE gốc | `bce, full_grid(no negative masking), pos_weight=2.0, field_weights, boundary_smoothing=0.1` + `[null] learnable null anchor enabled` | `set(m=0.5,λ_set=1.0,λ_fp=0.25,λ_lab=0.25,pool=s>0)` + class weight + BS 0.1 + null anchor |
| 3. Đủ 4 thư mục | ✅ | ✅ | ✅ |
| 4. Lọc 0.1 → 0.5 trùng decode 0.5 | 64/64 câu | 64/64 câu | 64/64 câu |

Phát hiện khi đọc code trong lúc dry test: `set` tắt negative masking còn BCE thì không, nên
bản `bce_bundle` đầu tiên **chưa** tách sạch được 3 term. Đã thêm `--full-grid` (có test trong
`tests/test_losses.py`) và chạy lại dry test cho `bce_bundle` (xem nhật ký).

Kiểm tra 4 mới chạy trên 32 câu mỗi split. Ở run thật, cả decode 0.5 lẫn dump 0.1 đều được
lưu, nên phải chạy lại phép so này trên toàn bộ test trước khi dùng kết quả của mục 6.1.

Chạy lại trên **RTX-PRO-6000** (`--gpu RTX-PRO-6000`, 96 GB): cả 3 recipe trên pii_500k và
`set` trên privy (bf16, 4×4) đều `status=ok`, không có OOM hay lỗi CUDA, kiểm tra 4 đạt 32/32
câu mỗi split. Thời gian 1.4–6.6 phút/run; 2 run đầu mất khoảng 5 phút khởi động container
lạnh trên loại GPU mới. Ở cỡ smoke, thời gian chủ yếu là overhead, nên chưa so được tốc độ
thật với A10G. Nếu cả vòng dùng một GPU, chạy với `--gpu RTX-PRO-6000 --round v2`.

### 9.1 Tín hiệu sớm từ pilot v1 (không cần GPU)

Quét threshold trong {0.5, 0.6, …, 0.9} trên prediction v1 (2 seed), chọn theo macro-F1 validation:

| SCL − BCE (trung bình 4 corpus) | macro | rare-half | micro |
|---|---|---|---|
| threshold cố định 0.5 | +0.71 | +1.57 | −0.14 |
| threshold chọn trên validation | +0.68 | +1.55 | −0.18 |

Như vậy BCE **không** bắt kịp chỉ nhờ nâng threshold. Vùng dưới 0.5 chưa kiểm tra được (bản dump
v1 chỉ giữ span có score ≥ 0.5); đó là việc của Giai đoạn 1.

## 9.2 Kết quả đợt 1 (2026-09-27, `scripts/rare_pii_tables.py`, 5000 resample)

Seed 42/1337, threshold tuned trong {0.50, …, 0.90}, trung bình 4 corpus, CI 95% bootstrap theo câu test:

| Phép so | Δ rare-half | Δ macro | Δ micro |
|---|---|---|---|
| (b) − (a) | +1.09 [+0.78, +1.47] | +0.46 | −0.15 |
| **(c) − (b)** (endpoint chính) | **+0.45 [−0.10, +0.78]**, p = 0.23 | +0.22 | −0.02 |
| (c) − (a) | +1.54 [+0.90, +1.98] | +0.68 | −0.17 |

**Theo luật ở mục 7: kết cục B.** Lợi ích của 3 term cạnh tranh chưa được xác lập (CI chứa 0).

Ghi chú bắt buộc đi kèm:
- (c) − (b) theo corpus: Privy +0.17, OpenPII +0.52, Kiji −0.08, Gretel +1.19.
- Trên Gretel, kết quả phụ thuộc seed: hai seed của (c) cho 90.05 và 87.07, của (b) cho 87.52 và 87.22.
- CI chỉ đo dao động do lấy mẫu câu test, không đo dao động giữa các seed.
- Chi phí ước tính: <workspace> khoảng $13.9, <workspace> khoảng $12.9.

## 9.3 Phân tích khám phá theo nhãn (ghi trước khi tính, 2026-09-27)

Đây là phân tích **khám phá**, làm sau khi đã thấy kết quả gộp. Nó không thay endpoint chính.

- **Giả thuyết chính (duy nhất):** trên các nhãn hiếm, Δ = F1(c) − F1(b) ở threshold tuned trên test, lấy trung bình theo seed, **tương quan dương** với tỉ lệ lỗi *sai loại* của (b) trên **validation** ở p0 = 0.5. Ý nghĩa: type ranking giúp nơi còn cạnh tranh giữa các loại.
- **Kiểm định:** Spearman trên 54 nhãn hiếm gộp 4 corpus, hai phía, α = 0.05. Kèm theo: dấu của tương quan trên từng corpus, và Mann–Whitney so nhóm trên/dưới trung vị tỉ lệ sai loại.
- **Chỉ mô tả, không dùng để kết luận:** tương quan với tỉ lệ lỗi sai biên, dự đoán thừa, bỏ sót, F1 của (b) và support. Phép so tương tự cho (b) − (a) với lỗi của (a).
- **Lý do đo biến giải thích trên validation:** tránh hồi quy về trung bình, vì Δ đo trên test.

**Kết quả** (`scripts/rare_label_analysis.py`, seed 42/1337, 53 nhãn hiếm; một nhãn không có gold trên validation):

- **Giả thuyết chính không được ủng hộ:** Spearman(Δ, tỉ lệ sai loại của (b)) = +0.18, p = 0.19.
  - Dấu của tương quan không nhất quán giữa các corpus: Privy +0.72, OpenPII −0.01, Kiji −0.43, Gretel +0.14.
  - So nhóm trên/dưới trung vị: +1.39 so với −0.20, Mann–Whitney p = 0.38.
- **Phát hiện chính:** toàn bộ mức tăng trung bình của (c)−(b) đến từ **một nhãn**, `PIN` của Gretel.
  - Nhãn này chỉ có **2 mẫu trong test**. Seed 42 của (c) đúng 1/2 (F1 66.7); mọi run còn lại của (a), (b), (c) đều 0/2.
  - Nhãn này cho Δ = +33.3. Tổng Δ của cả 53 nhãn là +30.7, tức 52 nhãn còn lại cộng lại khoảng −2.6.
  - Bỏ `PIN` thì Gretel (c)−(b) = −0.41 (có `PIN` là +1.19), và trung bình 4 corpus khoảng **+0.05**.
- **Chỉ mô tả:** (c)−(b) tương quan âm với tỉ lệ sai biên của (b), ρ = −0.35, p = 0.010. Với 6 biến được thử, con số này không qua được hiệu chỉnh Bonferroni (ngưỡng 0.0083).
- **(b)−(a) cải thiện trải đều:** 38/53 nhãn tăng, trung vị +0.36. Mức tăng lớn hơn ở những nhãn mà (a) bỏ sót nhiều (ρ = +0.33, p = 0.017) và có F1 thấp (ρ = −0.39, p = 0.004). Đây cũng chỉ là mô tả.
- **Hệ quả về phương pháp:** rare-half macro-F1 rất nhạy với các nhãn chỉ có vài mẫu test. Cần báo cáo thêm một phân tích độ nhạy có ngưỡng support tối thiểu trên test.

## 9.4 Audit dữ liệu 10 corpus PII (khám phá, 2026-09-27)

Đây là phân tích **khám phá**. Nó được thêm vào sau khi đã có kết quả đợt 1, chỉ đo trên dữ liệu (CPU, không train) và không thay endpoint hay luật quyết định ở mục 7.

- **Công cụ:** `scripts/pii_audit.py` ghi kết quả vào `out_pii_baselines/pii_audit.json`. Bảng gộp ở `pii_audit_table.md` (tạo bằng `--table`).
- **Phạm vi:** 4 corpus của protocol, cộng Nemotron-PII (repo chính thức), ai4privacy 300k và 400k, Gretel finance, TAB và MAPA. Các corpus ngoài protocol được audit **đúng như bản phát hành**. Tập test là split mà bản phát hành gọi là validation hoặc test.
- **Các probe:**
  - O: offset không khớp value.
  - S: support.
  - L: template hoặc value xuất hiện trong train.
  - C: cùng một chuỗi mang nhiều nhãn.
  - U: PII được regex/checksum xác nhận nhưng không có gold. Tách thành loại có trong schema và loại ngoài schema.
  - G: value là null hoặc markup, hoặc text còn chừa slot chưa điền.
  - B: span chồng nhau.
  - IAA: độ đồng thuận giữa các annotator của TAB.
- **Chưa có:** độ chính xác của từng probe (chưa kiểm tay), và việc chấm lại để xem kết luận giữa các arm có đổi không. Trước khi kiểm tay, mọi con số chỉ là **cận trên**. Ví dụ C của TAB gồm cả hoán dụ hợp lệ, như "United Kingdom" khi là ORG và khi là LOC.

## 9.5 Độ nhạy của kết luận trước các lỗi benchmark (ghi trước khi tính, 2026-09-27)

Đây là phân tích **khám phá**. Nó chấm lại các prediction đã có trên 4 corpus của protocol, chạy CPU, không train thêm.

- **Giữ nguyên:** threshold tuned của từng run giống bảng chính, không tune lại. Nhờ vậy, mọi thay đổi chỉ đến từ cách chấm.
- **Seed:** mỗi phép so dùng những seed mà cả hai arm cùng có.
- **Tool:** `scripts/pii_sensitivity.py`.

Các biến thể:

- **V0, gốc:** phải tái lập đúng bảng chính. Đây là kiểm tra tính đúng của script.
- **V1, ngưỡng support:** bỏ các nhãn có dưới k mention trong test, với k ∈ {10, 20, 50}. Nửa hiếm được tính lại trên các nhãn còn lại.
- **V2, chỉ value mới:**
  - Bỏ các gold mention có (nhãn, value viết thường) đã xuất hiện trong train.
  - Bỏ mọi prediction chồng lên các gold bị bỏ, bất kể nhãn của prediction.
  - Giữ tập nhãn hiếm của V0.
- **V3, gộp nhãn dễ lẫn:**
  - Tính n_AB = Σ_v min(n_A(v), n_B(v)) trên train.
  - Gộp cặp (A, B) nếu n_AB ≥ 5% số mention train của nhãn nhỏ hơn. Gộp theo thành phần liên thông.
  - Nửa hiếm được tính lại trên tập nhãn đã gộp.
- **V4, bỏ vùng bị cờ:** bỏ các prediction chồng lên PII đã được regex/checksum xác nhận mà không có gold (probe U), hoặc chồng lên slot chưa điền (probe G).

Luật đọc kết quả, cố định trước khi tính:

- **Kết luận (b) > (a) đứng vững** nếu ở **mọi** biến thể: Δ rare-half > 0 và p sign-flip < 0.05.
- **Kết luận "(c) không hơn (b) một cách chắc chắn" đứng vững** nếu ở mọi biến thể: p ≥ 0.05 hoặc Δ ≤ 0.
- Báo cáo mọi biến thể, kể cả khi kết quả bất lợi.

**Kết quả** (`out_pii_baselines/pii_sensitivity.md`, seed 42/1337):

- **V0 tái lập đúng bảng chính.**
  - Rare-half: 86.10 / 87.19 / 87.64.
  - (b)−(a) = +1.09 (p 0.016); (c)−(b) = +0.45 (p 0.117).
- **Cả hai kết luận đứng vững ở mọi biến thể.**
  - (b)−(a) nằm trong khoảng +0.87 đến +1.46, với p ≤ 0.031.
  - (c)−(b) nằm trong khoảng +0.05 đến +0.45, với p ≥ 0.117.
- **Mức tuyệt đối dịch chuyển nhiều hơn mọi khác biệt giữa các arm:**
  - **V1-10:** rare-half tăng khoảng +5 điểm, từ 86.1 lên 91.2. Mức tăng này chỉ do bỏ 4 nhãn có dưới 10 mention test (2 nhãn của Kiji, 2 nhãn của Gretel). Ở biến thể này (c)−(b) chỉ còn +0.05, khớp với phát hiện về `PIN` ở mục 9.3.
  - **V2:** rare-half giảm khoảng −3.3 điểm. Kiji có 32.464 trên 38.785 gold mention là value đã gặp trong train. Trên value mới, (b)−(a) tăng lên +1.46.
  - **V3:** gộp nhãn làm rare-half tăng khoảng +4.5 điểm. Luật 5% gộp mạnh ở Kiji, tạo ra một nhóm 7 nhãn tên/tổ chức/địa danh và một nhóm 6 nhãn giấy tờ. Ở OpenPII nó gộp đúng cặp SEX/GENDER.
  - **V4:** gần như không đổi. 4 corpus này ít vùng bị cờ; lỗi nặng nằm ở ai4privacy 300k/400k.

## 9.6 (b) với seed 2024 (đợt 2, 2026-09-27)

Seed 2024 của (b) đã có đủ trên cả 4 corpus, gồm Privy, bf16. Record `status=ok`, cờ train đúng recipe.

- **(b) rare-half tuned với 3 seed: 87.18**, so với 87.20 khi chỉ có 2 seed.
- Theo từng corpus:
  - Privy: 99.24 / 99.44 / 99.51.
  - OpenPII: 81.82 / 81.77 / 81.89.
  - Kiji: 80.28 / 80.28 / 80.31.
  - Gretel: 87.52 / 87.22 / 86.88.
- Độ lệch chuẩn giữa các seed rất nhỏ: từ ±0.02 (Kiji) đến ±0.32 (Gretel). Để so sánh, (c) trên Gretel lệch 3 điểm giữa 2 seed.
- **Phép so không đổi:** sign-flip và Wilcoxon chỉ dùng seed chung của hai arm, tức 42/1337.
  - Bootstrap (b)−(a) cho +1.08, CI [+0.77, +1.43]. Ở đây mỗi arm lấy trung bình trên các seed của riêng nó.

## 9.7 Zero-shot trên 10 corpus (ghi trước khi chạy, 2026-09-27)

Đây là phân tích **khám phá**. Mục đích là đo hệ quả của audit trên cả 6 corpus chưa có model nào được train, với chi phí GPU thấp vì chỉ chạy inference.

- **Dữ liệu:**
  - 6 corpus mới được chuyển bằng `scripts/prepare_pii_datasets.py`, dùng đúng quy trình của 4 corpus cũ:
    - Giới hạn 50k/2.5k/5k.
    - Xáo trộn với seed 42.
    - Mọi span phải thỏa `text[start:end] == value`; span không thỏa bị bỏ và được đếm.
    - Bỏ các nhãn có dưới 20 mention trong train.
  - Kết quả được đẩy lên repo private `anonymous-org/pii-*`.
  - Tập test là split test hoặc validation của bản phát hành, lấy tối đa 5k.
  - TAB không giới hạn độ dài train. MAPA chỉ có test.
- **Model:** `fastino/gliner2-multi-v1`, không train.
  - Schema và mô tả nhãn lấy từ registry.
  - Dùng `--use-desc --schema-case original`.
  - Decode ở threshold 0.5.
  - **Sửa trước khi chạy (cùng ngày):** chỉ decode test ở 0.5, bỏ validation và bản dump 0.1.
    - Lý do: smoke trên TAB cho khoảng 4,6 văn bản/giây, nên chạy đủ sẽ tốn khoảng $7–8, vượt $6.26 của workspace <workspace>.
    - Ba phân tích bên dưới chỉ cần test ở 0.5. Tune threshold không nằm trong 9.7.
  - Seed ghi là 0.
- **Phân tích, cố định trước:**
  1. **Recall trên value đã gặp so với value mới**, trên 8 corpus có train.
     - Model zero-shot chưa từng thấy train, nên nếu vẫn có khoảng cách thì khoảng cách đó đến từ độ khó của value, không phải do học thuộc.
     - So sánh với khoảng cách của (b) đã train trên 4 corpus protocol.
  2. **Precision khi có và khi bỏ các prediction nằm trong vùng bị cờ** (V4), trên cả 10 corpus. Quan trọng nhất là AI4Privacy-300k/400k.
  3. **Rare-half macro-F1 ở cách chấm gốc so với khi bỏ nhãn có dưới 10 mẫu test** (V1-10), trên từng corpus.
- **Không dùng để so với các run đã train.** Model zero-shot không được tune threshold. Số zero-shot chỉ được đọc bên trong từng corpus.
- **Sự cố khi chạy (ghi trước khi đọc kết quả của corpus đó):**
  - Trên A10G, Gretel-Finance bị CUDA OOM ở 4.900/5.000 tài liệu, trong phần inference theo batch 8 của `evaluate.py`.
  - `evaluate.py` ghi lỗi của từng mẫu thành một prediction rỗng mà vẫn trả `status=ok`, nên recall bị kéo xuống khoảng 1%.
  - Cách xử lý:
    - Chạy lại đúng cấu hình đó trên A100-40GB. Bản lỗi được giữ ở `v1-zeroshot/gliner2/gretel_finance.oom-a10g/`.
    - Từ nay runner đánh dấu `status=error` khi có bất kỳ mẫu nào lỗi.
  - Đã quét mọi dump khác trong `out_pii_baselines/`, gồm toàn bộ các run (a)/(b)/(c) của paper: **không có mẫu lỗi nào**.
  - Chạy lại trên A100 và với batch 1 vẫn OOM. Nguyên nhân: `evaluate.py` gom mọi mẫu sau 100 mẫu đầu vào **một** lệnh `batch_extract`, nên một chunk lỗi là toàn bộ đều lỗi.
  - Lần chạy thứ tư dùng `--infer-batch-size 0`: mọi mẫu chạy tuần tự, mỗi mẫu có try/except riêng.
    - Kết quả: chỉ còn **1/5.000 tài liệu** lỗi OOM, một văn bản tiếng Thụy Điển dài 5.271 ký tự với 2 span gold.
    - Tài liệu này bị **loại khỏi phần chấm**, và `docs_failed_excluded` ghi lại việc đó. Không tính nó thành prediction rỗng.

**Kết quả, đủ 10/10 corpus** (`out_pii_baselines/pii_zeroshot_analysis.json`). Nemotron-PII, Gretel-Finance và MAPA không làm đổi bất kỳ kết luận nào dưới đây:

- **Phân tích 1, đo gộp:**
  - Hiệu (b) − zero-shot của khoảng cách recall "đã gặp − mới" dương trên cả 4 corpus: Privy +7.3, OpenPII +1.4, Kiji +12.4, Gretel-PII +10.4.
    - Số này gộp cả 3 seed của (b).
    - Bản đầu chỉ dùng seed 1337: script chọn "seed đầu tiên" bằng cách sắp xếp tên như chuỗi ký tự, nên seed-1337 đứng trước seed-42. Khi đó số là +7.4 / +1.9 / +12.5 / +10.3; kết luận không đổi.
  - **Kiểm tra độ vững (post hoc, thêm sau khi thấy số gộp):** tính cùng hiệu đó **bên trong từng loại nhãn**, chỉ lấy loại có ít nhất 20 mention đã gặp và 20 mention mới.
    - Privy: +9.7 (trọng số đều) / +5.9 (trọng số theo mention), 3/6 loại dương.
    - Kiji: +2.4 / +3.7, 11/24 loại dương.
    - OpenPII: −5.4 / −8.6, 1/9 loại dương.
    - Gretel-PII: −10.8 / −4.6, 0/5 loại dương.
    - Số trên gộp 3 seed của (b).
  - ⇒ Số gộp **chủ yếu là hiệu ứng thành phần**: value đã gặp dồn vào những loại nhãn mà model đã train làm tốt. **Không được viết** là "train làm model học thuộc value". Chỉ Privy, và ở mức yếu là Kiji, còn dương bên trong từng loại.
- **Phân tích 2:** tỉ lệ FP nằm trong vùng bị cờ ≤ 0.3% ở mọi corpus, trừ AI4Privacy-400k với 3.68%. Precision của 400k đổi từ 38.56 lên 39.45 khi bỏ các FP đó.
- **Phân tích 3:** bỏ nhãn có dưới 10 mẫu test làm rare-half zero-shot tăng +7.5 ở Kiji và +3.0 ở Gretel-PII. Các corpus khác không có nhãn nào dưới 10 mẫu.

## 9.8 Train (a), (b) trên hai corpus có artifact nặng (ghi trước khi chạy, 2026-09-27)

Mục này trả lời nhận xét "does it matter chỉ được kiểm ở nơi ít lỗi".

- **Thiết kế:**
  - Arm (a) `bce` và (b) `bce_bundle`, dùng đúng recipe của mục 4.
  - Corpus: AI4Privacy-300k và AI4Privacy-400k (bản chuyển đổi ở 9.7). Seed 42 và 1337.
  - Cấu hình: A10G, batch 8×2, fp16, 3 epoch. Workspace <workspace>.
  - Decode validation và test ở 0.5. Threshold tuned lấy trên lưới {0.50, …, 0.90} theo macro-F1 validation, giống mục 6.1.
- **Phân tích:**
  - Rare-half, macro, micro.
  - Các biến thể V0–V4 của 9.5 trên hai corpus này, giữ nguyên threshold tuned. Đây là nơi V4 (vùng bị cờ) thực sự có dữ liệu.
- **Luật đọc kết quả:**
  - Có 2 corpus × 2 seed = 4 hiệu. Sign-flip vì thế có p nhỏ nhất là 0.125, nên **không kết luận về ý nghĩa thống kê**.
  - Báo cáo Δ rare-half (b)−(a) và số run dương, cho từng biến thể.
  - "(b) > (a) lặp lại" nếu Δ > 0 ở ít nhất 3/4 run dưới V0.
  - "Thứ hạng đứng vững" nếu dấu của Δ trung bình không đổi ở mọi biến thể.
  - Báo cáo thêm: V4 làm Δ thay đổi bao nhiêu.

## 9.9 Ablation: (a) + full grid (ghi trước khi chạy, 2026-09-27)

Mục này trả lời nhận xét "(b) là một bundle chưa ablate".

- **Arm (a′):** `--loss bce --full-grid`, tức (a) tắt negative masking và không bật gì khác của (b).
- **Thiết kế:**
  - 4 corpus case study, **chỉ seed 42** vì ngân sách.
  - GPU, batch và dtype theo từng corpus, giống hệt (a) và (b).
- **Đại lượng:** tỉ lệ (a′)−(a) chia (b)−(a), tính trên rare-half tuned, seed 42, cho từng corpus và gộp 4 corpus. Chỉ mô tả, vì chỉ có 1 seed.
- **Luật đọc:** "tắt masking giải thích phần lớn mức tăng" nếu hiệu gộp (a′)−(a) ≥ 0.5 × (b)−(a). Báo cáo dù kết quả ra sao.

## 9.10 Hệ thống dựng sẵn và độ ổn định của thứ hạng (ghi trước khi chạy, 2026-09-27)

Mục này trả lời nhận xét "headline yếu: thứ hạng giữa các hệ thống có đảo không".

- **Hệ thống:** chỉ chạy inference, không train.
  - GLiNER2 Multi-v1 zero-shot (đã có từ 9.7).
  - `fastino/gliner2-privacy-filter-PII-multi` (GLiNER2-PII). Train trên 4.910 văn bản tổng hợp của riêng họ.
  - `urchade/gliner_multi_pii-v1`. Train trên `urchade/synthetic-pii-ner-mistral-v1`.
  - `nvidia/gliner-PII`. **Train trên Nemotron-PII.**
  - `gretelai/gretel-gliner-bi-large-v1.0`. **Train trên split train của Gretel-PII.**
  - Thông tin dữ liệu train lấy từ model card, đọc ngày 2026-09-27.
- **Đầu vào:**
  - Test của 10 corpus, bản chuyển đổi ≤5k tài liệu. Threshold 0.5 cho mọi hệ thống, không tune.
  - Model GLiNER2 nhận tên nhãn cùng mô tả trong registry, giống 9.7.
  - Model GLiNER v1 nhận **mô tả trong registry làm chuỗi nhãn**, rồi ánh xạ ngược về loại. Văn bản được cắt thành đoạn ≤200 từ, nhãn được chia thành nhóm ≤20. Sau đó decode phẳng tham lam theo điểm.
  - Lỗi nào xảy ra ở từng mẫu thì run thất bại (guard của 9.7).
- **Endpoint:**
  - Trên mỗi corpus, xếp hạng các hệ thống theo rare-half macro-F1 dưới V0, V1-10, V2, V3 và V4.
  - V2 và V3 dùng train đã chuyển đổi, nên không áp dụng cho MAPA.
  - Báo cáo Kendall τ giữa V0 và từng biến thể, và hệ thống đứng đầu có đổi không. Đếm số corpus có hệ thống đứng đầu đổi, theo từng biến thể.
- **Giả thuyết phụ, ghi trước:** mỗi model được train trên một corpus (nvidia → Nemotron-PII, gretel → Gretel-PII) sẽ xếp hạng cao hơn trên chính corpus đó so với trung vị hạng của nó trên các corpus còn lại. Báo cáo hạng thực tế.
- **Không kiểm định ý nghĩa:** mỗi hệ thống chỉ có 1 run, nên mọi kết quả là mô tả.

### 9.8–9.10: sai lệch và kết quả (2026-09-27)

**9.8: AI4Privacy-300k bị hủy. Sai lệch này được ghi trước khi đọc bất kỳ kết quả nào của 400k.**

- Trên A10G, batch 8×2, cả 4 run 300k ((a) và (b) × seed 42/1337) báo `OOM at step N, batch skipped`, khoảng **94–135 batch mỗi run**.
- Trainer bỏ qua các batch đó, tức là âm thầm bỏ đúng những tài liệu nặng nhất. Theo luật của dự án, những run này không được dùng.
- Hai run seed 1337 còn chết ở phút thứ 80 rồi bị Modal chạy lại.
- Workspace <workspace> đã tiêu $26.86/$30. Mức tiêu cao hơn ước tính vì phí CPU và RAM, khoảng $0.44/giờ mỗi container.
- Để không chạm hạn mức (chạm là mọi job bị giết, kể cả (b) 400k đang hợp lệ), cả 4 run 300k đã bị **hủy** (`cancel(terminate_containers=True)`).
- Chạy lại trên A100 với batch 4×4 tốn khoảng $15, vượt ngân sách còn lại. Vì vậy **9.8 chỉ còn AI4Privacy-400k**: 2 seed × 2 arm, tức 2 hiệu.
  - Container của các run 400k có **0** batch bị bỏ.
  - Luật "≥ 3/4 run dương" được thay bằng "2/2 run dương", và vẫn chỉ đọc chiều của Δ.

**9.9: kết quả** (`pii_ablation_fullgrid.json`, seed 42, rare-half tuned):

| Corpus | (a′)−(a) | (b)−(a) |
|---|---|---|
| Privy | +0.16 | −0.11 |
| OpenPII | **−3.07** | +2.16 |
| Kiji | +0.16 | +0.27 |
| Gretel-PII | +0.90 | +2.49 |
| **Gộp 4 corpus** | **−0.46** | **+1.20** |

- ⇒ Luật đọc cho kết quả **"không"**: tắt negative masking một mình **không** giải thích mức tăng của (b). Ở OpenPII, tắt masking còn làm điểm giảm.
- Phần tăng vì vậy đến từ các thành phần còn lại của (b) (class weight, pos weight, boundary smoothing, null anchor), hoặc từ tương tác giữa chúng với full grid. 1 seed chưa đủ để tách tiếp.

**9.10: sự cố khi chạy.**

- 10 job Gretel GLiNER trên <workspace> lỗi ngay lúc khởi động (`recipe 'zeroshot' has no gliner variant`). Các call đã chạy bằng bản deploy cũ. Chúng đã được chạy lại trên <workspace> với cùng cấu hình.
- GLiNER2-PII trên Gretel-Finance lỗi OOM ở 1 tài liệu, chính là tài liệu tiếng Thụy Điển 5.271 ký tự của 9.7. Tài liệu này bị loại khỏi phần chấm và có ghi lại.

**9.8: kết quả, AI4Privacy-400k** (`pii_sensitivity_ai4p.json`):

- (b)−(a) rare-half = **+0.46**, 2/2 run dương (+0.30 và +0.62).
- Dấu không đổi ở mọi biến thể. V2 cho +0.49. V1 và V3 không đổi gì, vì 400k không có nhãn nào dưới 50 mẫu test và không có cặp nhãn nào bị gộp.
- **V4 làm Δ thay đổi 0.000:** model đã train có **0** prediction trong vùng bị cờ (URL, IP, slot chưa điền). Model zero-shot GLiNER2 thì có 189 FP ở đúng những vùng đó.
  - ⇒ Train trên corpus làm model học luôn chỗ trống của schema. Artifact không bao giờ phạt các model cùng họ, chỉ phạt những hệ thống không được train trên corpus đó.

**9.10: kết quả** (`pii_offshelf_ranking.json`, 5 hệ thống):

- **Thứ hạng ổn định qua các biến thể:** τ trung vị = 1.0 ở V1-10, V2, V3 và V4.
- **Hệ thống đứng đầu đổi ở 0/10 corpus** dưới V1-10 và V4, 0/9 dưới V3, và **2/9 dưới V2** (AI4Privacy-300k, TAB).
- **Hệ thống đứng đầu theo corpus (V0):**
  - GLiNER multi-PII: Privy, AI4Privacy-400k, MAPA.
  - GLiNER2-PII: OpenPII, Kiji, TAB.
  - Gretel GLiNER: Gretel-PII.
  - NVIDIA: Nemotron-PII, Gretel-Finance.
  - GLiNER2 Multi-v1: AI4Privacy-300k.
  - ⇒ 5 hệ thống khác nhau đứng đầu trên 10 corpus.
- **Giả thuyết phụ được ủng hộ:**
  - Gretel GLiNER đứng **hạng 1/5 trên Gretel-PII** (bộ nó được train), trong khi hạng trung vị ở 9 corpus còn lại là **5/5**.
  - NVIDIA đứng hạng 1/5 trên Nemotron-PII (bộ nó được train), hạng trung vị ở nơi khác là 2.
- **Đính chính bản xem trước (4/5 hệ thống) đã báo cho nhóm tác giả:** dòng Privy bị cắt khỏi output. Bản xem trước ghi GLiNER2-PII thắng Privy; thực ra hệ thống đó thắng OpenPII.

## 9.11 Ma trận chuyển giao có kiểm soát (ghi trước khi chạy, 2026-10-01)

Mục này kiểm chặt luận điểm "hệ thống được train trên corpus nào thì thắng corpus đó". Ở 9.10 luận điểm này chỉ dựa trên 2 model dựng sẵn, khác nhau ở cả kiến trúc lẫn dữ liệu.

- **Model:** objective (b), seed 42, mỗi model train trên **một** corpus với cùng code, recipe và cấu hình.
  - Privy, OpenPII, Kiji, Gretel-PII: dùng checkpoint đã giữ từ đợt 1.
  - AI4Privacy-400k: train lại seed 42 với `keep_ckpt`. Bản đã dùng ở 9.8 không giữ checkpoint; bản train lại có thể khác bit nhưng cùng cấu hình.
  - AI4Privacy-300k: lấy (b) seed 42 của 9.12, nếu run đó hợp lệ.
  - Checkpoint được đẩy lên repo private `anonymous-org/pii-ckpt-bundle-<ds>-s42` để chạy ở workspace khác.
- **Đánh giá:**
  - Mỗi model chấm trên test của cả 10 corpus, threshold 0.5.
  - Schema và mô tả là của corpus đích (GLiNER2 nhận nhãn tùy ý). Dùng đúng lệnh `evaluate.py` như 9.7.
  - Gretel-Finance chạy với `--infer-batch-size 0`.
- **Endpoint:** rare-half macro-F1. Xếp hạng trên mỗi corpus gồm các model (b), GLiNER2 Multi-v1 zero-shot và 4 hệ thống dựng sẵn của 9.10.
- **Giả thuyết, ghi trước:**
  - (H1) Mỗi model (b) đứng **hạng 1** trên corpus nó được train.
  - (H2) Ở các corpus còn lại, hạng trung vị của nó thấp hơn hạng ở nhà.
  - (H3) Báo cáo thêm, không có giả thuyết hướng: model (b) hơn hay kém GLiNER2 zero-shot trên các corpus khác (train trên một corpus tổng hợp làm tăng hay giảm khả năng chuyển giao).
- **Không có kiểm định ý nghĩa:** 1 seed, chỉ mô tả.

**9.11, phần bổ sung (ghi trước khi chạy, 2026-10-01, sau khi 9.11 đã khởi chạy nhưng trước khi có kết quả):**

- Thêm hai home: **Nemotron-PII** và **Gretel-Finance**.
- Mỗi home là một run (b) seed 42 có `keep_ckpt`, chạy trên A100-40GB với batch 4×4 (như Gretel-PII), ở workspace <workspace> (profile "<workspace>").
- Đánh giá giống hệt 9.11.
- H1–H3 áp dụng thêm cho hai model này.
- So sánh mô tả thêm: (b)-Nemotron với NVIDIA GLiNER-PII trên Nemotron-PII (hai model khác nhau, cùng được train trên corpus đó).

**9.11, sự cố (2026-10-01):**

- Checkpoint Privy và OpenPII được tải từ volume của "<workspace>" về máy local rồi đẩy lên HF, vì workspace đó không build được image để đẩy thẳng. Kiểm tra tính đúng (chấm lại trên corpus nhà phải tái lập dump gốc) cho thấy:
  - Kiji và Gretel-PII, đẩy thẳng từ Modal: **0** tài liệu lệch.
  - Privy và OpenPII: 106 và 462 tài liệu lệch.
- Tải lại cùng một file cho ra **hash khác**: lần tải local đã hỏng âm thầm mà kích thước vẫn đúng.
- 20 kết quả chấm bằng hai checkpoint này bị cách ly ở `*.corrupt-download/` và **không dùng**.
- Cách sửa:
  - Tải lại cho đến khi hai lần liên tiếp có cùng hash, rồi đẩy lên HF.
  - Chạy trước hai job chấm trên corpus nhà. Chỉ chạy 18 job còn lại khi corpus nhà tái lập dump gốc với 0 tài liệu lệch.
- Mọi checkpoint khác được đẩy thẳng từ Modal (`push_checkpoint`).

**9.11, Gretel-Finance (2026-10-01, trước khi có kết quả nào của home này):**

- (b) seed 42 trên A100-40GB với batch 4×4 bị guard dừng ở batch bị bỏ đầu tiên, sau khoảng 1,5 giờ train.
- Modal tự chạy lại; lần chạy lại đã bị hủy.
- Run được chạy lại trên **A100-80GB với batch 4×4**, như cách đã làm với Privy.
- Từ giờ một lần guard dừng không kích hoạt chạy lại nữa: runner ghi `status=error` và kết thúc mà không raise.
- **Nemotron-PII:** cũng bị guard dừng trên A100-40GB, sau khoảng 2 giờ. Run được chạy lại trên A100-80GB với batch 4×4, cùng cách xử lý như trên.

## 9.14 Chẩn đoán batch bị bỏ trong cấu hình gốc (ghi trước khi chạy, 2026-10-01)

- **Lý do:** ablation (a′) seed 1337 trên OpenPII (A10G, 8×2) bị guard dừng ở batch bị bỏ đầu tiên. Các run (b) của bảng chính trên OpenPII và Kiji cũng chạy A10G 8×2 với full grid, nhưng **khi đó chưa có guard**, nên có thể đã lặng lẽ bỏ batch.
- **Phép đo:**
  - Chạy lại (b) seed 42 trên OpenPII và Kiji, đúng cấu hình gốc (A10G, 8×2, fp16), ở chế độ **đếm** (`count_skips`): không dừng, chỉ đếm số dòng `batch skipped` và `loss=nan`.
  - Cùng seed nghĩa là cùng thứ tự batch, nên số đếm phản ánh run gốc. Bộ cấp phát CUDA có thể khác chút ít; đây là một ước lượng.
- **Luật đọc:**
  - Nếu cả hai ra 0: kết quả chính không bị ảnh hưởng.
  - Nếu khác 0: báo số batch bị bỏ (trên tổng số bước) trong paper, như một giới hạn. Không sửa số đã báo.
- **Ablation (a′) seed 1337 trên OpenPII** chạy lại trên A100-40GB với batch 8×2 (chỉ đổi GPU), và có ghi lại điều này.

### Kết quả 9.11–9.14 (2026-10-01)

**9.11** (`pii_transfer_matrix.json`, 8 home × 10 corpus):

- **H1: đúng 8/8.** Mọi model (b) đứng hạng 1/13 trên corpus nhà.
- **H2: đúng.** Hạng trung vị ở các corpus khác nằm trong khoảng 3–10.
- **H3:** 6/8 model kém GLiNER2 chưa train trên các corpus khác (trung vị −13.6 đến −1.6). Hai ngoại lệ là Kiji (+3.3) và Nemotron-PII (+6.3).
- **Kiểm tra pipeline:** chấm lại trên corpus nhà tái lập dump gốc khớp tuyệt đối ở 8/8 model.
- (b)-Nemotron đạt 96.8 trên Nemotron-PII, NVIDIA GLiNER-PII đạt 84.0.

**9.12** (`pii_sensitivity_ai4p.json`, AI4Privacy-300k và 400k):

- (b)−(a) = +0.24, 4/4 run dương ⇒ "lặp lại".
- Dấu không đổi ở mọi biến thể. Ở V3 chỉ 2/4 run dương. V4 làm Δ thay đổi khoảng 0.

**9.13** (`pii_ablation_fullgrid.json`, seed 42 và 1337):

- (a′)−(a) = −0.09, so với (b)−(a) = +1.09 ⇒ **không**: masking không giải thích mức tăng.
- 7/8 run có (a′)−(a) dương nhưng nhỏ; OpenPII giảm −1.38.
- (a′) seed 1337 trên OpenPII chạy trên A100-40GB, vì trên A10G nó bỏ batch.

**9.14:** cấu hình gốc của (b) seed 42 bỏ **0** batch và có **0** dòng NaN, trên cả OpenPII lẫn Kiji.

## 9.12 Hoàn tất 9.8: AI4Privacy-300k trên A100 (ghi trước khi chạy, 2026-10-01)

- (a) và (b), seed 42 và 1337, chạy trên **A100-40GB với batch 4×4** (effective batch 16, giống Gretel-PII).
- Runner **dừng ngay** khi gặp `batch skipped` hoặc `loss=nan` (`_run_train`). Run bị dừng không được dùng. Nếu xảy ra, chạy lại trên A100-80GB và ghi sai lệch.
- Luật đọc giống 9.8. Gộp với 400k thì có 4 hiệu; "(b) > (a) lặp lại" nếu ít nhất 3/4 run dương.

## 9.13 Ablation (a′) seed 1337 (ghi trước khi chạy, 2026-10-01)

- (a′) = `--loss bce --full-grid`, seed 1337, trên 4 corpus case study.
- Đại lượng và luật đọc giống 9.9, nhưng tính trên 2 seed (42, 1337). Các hiệu có cặp với (a) và (b) cùng seed.

## 9.15 (b) seed 42 trên TAB, H200 (ghi trước khi chạy, 2026-10-01)

- **Mục đích:** điền ô TAB của phân tích 1 trong 9.7 (hiệu "đã gặp − mới" giữa model đã train và zero-shot, đo gộp và bên trong từng loại). Trước đây ô này trống vì chưa có model nào được train trên TAB.
- **Dữ liệu:** trainer của GLiNER2 cắt mọi văn bản ở `max_len = 2048` từ mà không báo. 200/1.014 bản án trong train dài hơn mức đó, nên sẽ mất 9.963/87.459 span train (11,4%).
  - `scripts/chunk_tab.py` cắt train và validation thành các đoạn ≤ 1.000 từ (tách theo khoảng trắng). Ưu tiên cắt ở xuống dòng và **không bao giờ cắt giữa một span**. Kết quả được đẩy lên `anonymous-org/pii-tab-chunked` (private, registry key `tab_chunked`).
  - Đã kiểm tra:
    - Số span không đổi: train 87.459 → 87.459, validation 7.916 → 7.916.
    - 0 offset lệch.
    - Đoạn dài nhất là 1.307 token của splitter, dưới 2.048.
  - **Test giữ nguyên**: 127 văn bản, 7.563 span. Phần chấm dùng `anonymous-org/pii-tab` như mọi hệ thống khác.
- **Run:** `--recipe bce_bundle`, seed 42, `--train-dataset tab_chunked`, eval `--datasets tab`.
  - H200, batch 4×4 (effective 16), fp16, 3 epoch, threshold 0.5.
  - Có guard: dừng ở batch bị bỏ đầu tiên hoặc `loss=nan`. Nếu xảy ra: chạy lại một lần với bf16 và ghi sai lệch.
  - Workspace `<workspace>`. Smoke trước, rồi mới chạy thật.
- **Đọc:** dùng nguyên `pii_zeroshot_analysis.py`, không đổi luật. Ngưỡng ≥ 20 mention đã gặp và ≥ 20 mention mới mỗi loại. Đây là 1 seed, còn các corpus khác gộp 2–3 seed; paper phải ghi rõ điều này.
- Không đưa TAB vào ma trận chuyển giao 9.11 trong đợt này.
- **Kết quả (2026-10-01):**
  - Run `status=ok` trên H200, mất 541 s cả train lẫn eval. Guard: 0 batch bị bỏ, 0 NaN. 0/127 tài liệu test lỗi.
  - Dump đã được tải hai lần và có hash trùng nhau.
  - Recall đã gặp/mới: zero-shot 61.2/42.6, (b) 83.3/64.3.
  - Hiệu (b) − zero-shot của khoảng cách đã gặp − mới:
    - Đo gộp: +0.4.
    - Bên trong loại, trên 8 loại: −0.7 (trọng số đều) / +3.0 (trọng số theo mention), 3/8 loại dương.
  - Tính cả 9 corpus có train: đo gộp dương 8/9; bên trong loại dương 5/9, âm 4/9.
  - Kết luận của 9.7 không đổi: phần lớn khoảng cách đo gộp là hiệu ứng thành phần.
  - Chi phí ước tính khoảng $0.9, gồm smoke 130 s và run thật 541 s, ở giá H200 cộng CPU/RAM.

## 9.16 Kiểm tra mẫu cờ audit bằng model thay cho người (ghi trước khi đọc kết quả, 2026-10-01)

- **Thay đổi:** không có người gán nhãn để check tay 435 mục của `audit_validation/sample.csv`, nên phần check do **model (Claude) làm**.
  - Paper phải gọi đây là **model review**, không gọi là hand check.
  - Không báo cáo κ giữa người với người.
- **Cách làm:**
  - 3 lượt độc lập. Mỗi lượt là một agent riêng, không thấy kết quả của lượt khác.
  - Đầu vào giấu tên corpus. Thứ tự mục được xáo khác nhau mỗi lượt (seed 11/22/33).
  - Câu hỏi và tập đáp án đóng giữ đúng như sheet: O H/B/?, C E/L/?, U M/S/N/?, G A/T/?.
  - Agent không được chạy code hay regex để gán nhãn.
  - Kết quả lưu ở `audit_validation/model_pass{1,2,3}.csv`.
- **Chấm** (`pii_audit_sample.py --passes`):
  - Nhãn của mỗi mục là nhãn được ít nhất 2/3 lượt chọn. Mục không có đa số, hoặc có đa số là `?`, bị loại.
  - "Vấn đề thật" = O:H, C:E, U:M, G:A. Báo cáo tỉ lệ đó kèm khoảng Wilson 95%.
  - Đồng thuận: Fleiss' κ, Cohen κ từng cặp lượt, và % mục cả 3 lượt trùng nhau.
  - Đây là **độ nhất quán của một model**, không phải đồng thuận giữa người gán nhãn, và paper phải ghi đúng như vậy.
- Luật đọc giữ nguyên với hand check: nhân tỉ lệ đã ước lượng với số cờ để ra cận trên đã hiệu chỉnh.
- **Mở rộng mẫu** (nhóm tác giả yêu cầu, cùng ngày). Ghi sau khi đã thấy kết quả 435 mục đầu nhưng trước khi gán nhãn phần mới:
  - Mỗi ô (corpus, probe) được bổ sung lên **tối đa 50 mục** bằng `pii_audit_sample.py --extend 50`.
  - Mục mới rút từ các cờ còn lại của ô, với seed `2026-<ds>-<probe>-ext`.
  - Kết quả: thêm 529 mục, tổng 964. Bản cũ lưu ở `sample_v1_435.csv`, phần mới ở `sample_ext.csv`.
  - Đáp án của 435 mục cũ giữ nguyên.
  - 529 mục mới được gán nhãn bởi 3 lượt mới, cùng quy trình: blind, giấu tên corpus, xáo seed 11/22/33. File kết quả `model_pass{k}_ext.csv`.
  - Lượt k của phần mới được nối với lượt k của phần cũ, rồi chấm trên cả 964 mục.
  - Paper báo cáo số trên 964 mục; số của 435 mục đầu giữ trong `validation_scores_v1_435.json`.
- **Kết quả trên 964 mục** (gán nhãn đa số, CI Wilson 95%):
  - G: 88.3 [83.5, 91.8]
  - O: 50.0 [40.4, 59.6]. AI4Privacy-300k 98%, Nemotron 2%.
  - C: 21.4 [17.9, 25.4]. MAPA 64%, Gretel-Finance 35%, TAB 32%.
  - U: 47.6 [40.2, 55.1] là PII thuộc schema; 83.9% là PII thật nói chung. URL của AI4Privacy-400k: 100% là PII thật ngoài schema.
  - Fleiss κ 0.84–1.00 theo probe; κ từng cặp lượt 0.79–1.00.
  - So với 435 mục đầu, tỉ lệ mỗi probe lệch tối đa 2.5 điểm.
- **Sự cố 2026-10-01 (chi phí):** khi deploy lên `<workspace>`, 4 call GLiNER v1 `--recipe set` (pii_500k và kiji, seed 42/1337, A100-40GB) đã chạy theo.
  - 4 call này được spawn từ tháng 9 qua profile `<workspace>`, dùng chung workspace `<workspace>`.
  - Hồi đó chúng bị kẹt vì workspace hết hạn mức.
  - Chúng chạy khoảng 27 phút rồi bị hủy theo yêu cầu nhóm tác giả; paper không dùng các run này.
  - Bài học: trước khi deploy lên một workspace, phải kiểm tra `get_current_stats()` và các call cũ của **mọi profile** trỏ vào workspace đó.

## 9.17 Control cho W2 của review: TAB làm home trong ma trận chuyển giao (ghi trước khi chạy, 2026-10-01)

- **Câu hỏi:** model train trên một corpus thì thắng corpus đó và tụt ở nơi khác. Đó là do *generator* synthetic, hay chỉ là fit với corpus train nói chung? Còn hai khả năng nữa: lệch tên nhãn/schema, và quên khả năng zero-shot.
- **Run:**
  - (b) seed 42 trên TAB, với `--train-dataset tab_chunked`, H200, cấu hình đúng như 9.15.
  - `--round tabhome`, tức tag `tabhome-bce-bundle-50k`, giữ checkpoint. Push checkpoint từ Modal lên `anonymous-org/pii-ckpt-bundle-tab-s42` (private).
  - Eval trên test của 10 corpus bằng recipe `zeroshot`, A10G, giống 9.11. Gretel-Finance dùng `--infer-batch-size 0`.
  - Kiểm tra pipeline: dump TAB khi re-eval phải trùng đúng dump test của chính run đó.
- **Đọc (cùng H1–H3 của 9.11):**
  - **H1:** model TAB đứng nhất trên TAB.
  - **H2:** hạng trung vị của nó ở nơi khác kém hơn ở nhà.
  - **H3:** báo cáo điểm ở nơi khác trừ điểm của model chưa train.
  - **Luật diễn giải, cố định trước:** nếu model TAB, một corpus do người gán, cũng thắng ở nhà **và** thấp hơn model chưa train trên đa số (≥ 5/9) corpus khác, thì paper **không được** quy hiện tượng cho "generator". Claim phải đổi thành "fit với corpus train (schema, giá trị, văn phong)". Nếu không, vẫn chỉ được nói là tương thích với giải thích generator, vì schema và miền của TAB khác hẳn.
- **Phân tích tách nguyên nhân, chỉ cần CPU trên dump có sẵn và dump mới:**
  - **D1, co cụm loại:** với mỗi model đã train × corpus đích (khác nhà), đếm số loại đích có 0 prediction, so với model chưa train trên cùng corpus.
  - **D2, tên nhãn:** với mỗi cặp, so recall của loại đích có tên trùng (không phân biệt hoa/thường) với một loại trong schema nhà, và recall của loại có tên lạ, ở model đã train và model chưa train.
    - Nếu mức tụt tập trung ở loại tên lạ hoặc loại 0 prediction, thì lệch schema/quên giải thích phần lớn.
  - **D3 (W8):** ranking của Table 6/7 tính lại bằng micro-F1 strict; báo cáo Kendall τ giữa ranking rare-half và ranking micro trên mỗi corpus.
- **Ngoài phạm vi:** train trên hỗn hợp corpus, và paraphrase nhãn. Hai việc này ghi vào Limitations.
- **Kết quả (2026-10-01):**
  - **Run:** train 473 s trên H200, guard sạch. Checkpoint được push từ Modal.
  - **Eval:** 10 job đều có dump, hash khớp sau hai lần tải. Gretel-Finance có 1 mẫu OOM, đúng văn bản 2663 như mọi lần eval trước; mẫu này bị loại nhất quán.
  - **Kiểm tra pipeline:** re-eval trên TAB cho 57.46, đúng bằng run gốc.
  - **H1 đúng:** hạng 1/14 trên TAB, 57.5 so với 29.8 của hệ thống đứng sau.
  - **H2 đúng:** hạng trung vị ở nơi khác là 14/14, tức đứng cuối.
  - **H3:** thấp hơn model chưa train trên 8/9 corpus khác (trung vị −23.8). Chỉ MAPA tăng (+20.7); MAPA cũng là văn bản pháp lý do người gán.
  - ⇒ Theo luật đã ghi trước: **không quy hiện tượng cho "generator"**. Claim đổi thành "fit với corpus train".
  - **D1/D2 (`pii_review_checks.py`):**
    - Trên 72 cặp (8 home synthetic × 9 đích), 43 cặp thấp hơn model chưa train. Chỉ 17/43 trong số đó có thêm loại 0 prediction.
    - Hiệu trung vị trên các loại cả hai model đều dự đoán là −3.8, so với −4.7 khi tính toàn bộ. Như vậy co cụm loại không phải nguyên nhân chính.
    - Recall trung vị trên loại **tên lạ**: 34.9 (đã train) so với 54.4 (chưa train). Trên loại **tên trùng**: 69.0 so với 74.8.
    - Model TAB cho cùng mẫu hình: tên lạ 31.6 so với 59.7; tên trùng 42.0 so với 50.7.
    - ⇒ Phần lớn mức tụt nằm ở các loại mà schema lúc train không có tên, phù hợp với giải thích chuyên biệt hóa schema / quên zero-shot (W2 a/b).
  - **W3:**
    - Trong 40 cặp hệ thống phát hành liền kề, 15 cặp cách nhau dưới 3 điểm. 8 trong 15 cặp đó đảo thứ hạng dưới ít nhất một variant; không cặp nào cách ≥ 3 điểm bị đảo. Cặp đảo có margin lớn nhất là 2.07.
    - Lấy corpus làm đơn vị (n = 4, p nhỏ nhất đạt được là 0.125): (b) > (a) trên 4/4 corpus ở mọi variant. (c) − (b) chỉ dương trên 2–3/4 corpus.
  - **W8:**
    - Kendall τ trung vị giữa ranking rare-half và ranking micro: 0.54 trên 13 hệ thống, 0.20 trên 5 hệ thống phát hành.
    - Hệ thống phát hành đứng đầu chỉ giữ nguyên trên 2/10 corpus khi đổi metric.
    - Model train tại nhà vẫn đứng nhất dưới micro trên 8/8 corpus.
- **W5 (thêm 2026-10-02, khám phá, `pii_review_checks.py` khối W5):** lặp lại phân tích W3 bằng micro-F1.
  - **Hệ thống phát hành:**
    - V1, V3, V4: τ trung vị 1.0; hệ thống đứng đầu đổi ở 0/10, 2/9, 0/10 corpus.
    - 12/20 cặp liền kề cách nhau dưới 3 điểm bị đảo.
    - **V2 còn đảo 2 cặp cách xa:** 21.0 điểm trên Kiji và 8.9 trên Gretel-Finance.
    - ⇒ Câu "chỉ đảo các cặp sát nhau" chỉ đúng hoàn toàn với rare-half; dưới micro-F1 thì V2 là ngoại lệ.
  - **Objective dưới micro-F1:** (b) − (a) = −0.15 (1/4 corpus dương) ở mọi variant, nên không hiệu chỉnh nào đổi thứ tự này.
  - **Phạm vi paper:** đổi tiêu đề phụ thành "with GLiNER Extractors" và thêm đoạn "Why GLiNER models". Lý do là thiết kế cần một model nhận schema lúc test; không lập luận rằng "chỉ GLiNER2 mới generalize".

## 9.18 Can thiệp tên nhãn: paraphrase lúc eval (ghi trước khi chạy, 2026-10-01)

- **Câu hỏi nhân quả:** 9.17 cho thấy, theo quan sát, mức tụt ở corpus khác dồn vào các loại có tên lạ. Câu hỏi ở đây: model đã train có **bám vào chuỗi tên nhãn** nó được train với không? Thí nghiệm đổi đúng một yếu tố là tên nhãn và giữ mọi thứ khác.
- **Can thiệp:** `evaluate.py --label-names description`.
  - Tên mỗi loại được thay bằng chính câu mô tả trong registry; mô tả giữ nguyên.
  - Prediction được map ngược về tên gốc trước khi chấm, với cùng scorer strict.
  - Đã kiểm tra trước: ở cả 9 schema, mô tả không trùng nhau và không trùng tên gốc.
- **Run:** test split, threshold 0.5, A10G (Gretel-Finance dùng `--infer-batch-size 0`), cấu hình còn lại giống 9.11.
  - **Model đã train:** 9 model (b) seed 42, 8 synthetic + TAB, mỗi model chỉ eval trên **corpus nhà** của nó.
  - **Đối chứng:** model chưa train (GLiNER2 Multi-v1) trên cùng 9 corpus đó.
  - Tag: `v1-zeroshot-labelpara*`.
- **Đại lượng:** với mỗi corpus h, Δ_trained = F1(paraphrase) − F1(tên gốc) của model train trên h, và Δ_untrained là đại lượng tương ứng của model chưa train. DiD = Δ_trained − Δ_untrained. Tính trên rare-half macro-F1 (chính) và micro-F1 (phụ).
- **Giả thuyết, luật đọc cố định trước:**
  - **P1:** DiD < 0 trên ≥ 7/9 corpus, tức model đã train mất nhiều hơn khi chỉ đổi tên nhãn. Nếu đúng, paper được viết "train trên một corpus làm model bám vào tên nhãn của schema đó", có bằng chứng can thiệp.
  - Nếu DiD ≥ 0 trên đa số corpus: claim về tên nhãn trong 5.4 bị **hạ** xuống mức quan sát, và paper phải nói rõ can thiệp không ủng hộ nó.
  - Báo cáo kèm kiểm định sign-flip chính xác trên 9 DiD; p nhỏ nhất đạt được là 2/512 ≈ 0.004.
- **Ngoài phạm vi:** đổi tên trên corpus khác (map sang tên của schema nhà), và paraphrase tự viết tay.
- **Kết quả (2026-10-01, `scripts/pii_labelpara.py`):**
  - Đủ 18 dump, mỗi dump tải hai lần và hash khớp. Gretel-Finance có 1 mẫu OOM (văn bản 2663), giống mọi run khác, bị loại ở cả bốn dump.
  - Δ_trained trung vị là **−0.04**: tệ nhất −2.16 (Gretel-Finance), có 3/9 corpus dương. Đổi tên gần như **không ảnh hưởng** model đã train trên corpus nhà.
  - Δ_untrained trung vị là **+3.06**, dao động từ −4.97 (Nemotron) đến +13.32 (Privy). Model chưa train thay đổi theo tên, phần lớn là tăng khi tên giàu thông tin hơn.
  - DiD < 0 trên **6/9** corpus, trung vị −3.02, sign-flip p = 0.082 ⇒ **P1 không đạt** (cần ≥ 7/9).
  - Micro-F1 (phụ): 7/9, p = 0.039.
  - **Diễn giải theo luật:** không được viết "train làm model bám vào tên nhãn". Kết quả đi theo chiều ngược lại: model đã train không phụ thuộc chuỗi tên, còn model chưa train thì có.
  - Mức tụt ở corpus khác (9.17 D2) giờ được hiểu là tụt ở **loại không có trong schema lúc train**, không phải ở tên lạ. Tên và loại trùng nhau gần như hoàn toàn trong phép tách D2, nên D2 không phân biệt được hai cách giải thích; can thiệp này phân biệt được.
  - Diễn giải "train làm mất sự phụ thuộc vào tên" là **khám phá**, không có trong đăng ký trước.

## 9.19 Đặt tên nhãn cho model chưa fine-tune: chọn trên validation (ghi trước khi chạy, 2026-10-01)

- **Câu hỏi thực dụng:** khi dùng một model GLiNER2 mà không fine-tune, nên đặt tên nhãn thế nào? 9.18 cho thấy dùng mô tả làm tên làm model chưa train dao động mạnh, từ −5.0 đến +13.3, và không có quy tắc đơn giản nào dự đoán được chiều. Vì vậy ở đây cách đặt tên được **chọn trên validation** và **đánh giá trên test**.
- **Ba cách đặt tên (deterministic, không tự viết tay):**
  - S0 `original`: tên trong registry.
  - S1 `humanized`: chữ thường, `_` thành khoảng trắng.
  - S2 `description`: mô tả trong registry dùng làm tên.
  - Ở cả ba cách, mô tả giữ nguyên và `--use-desc` luôn bật.
- **Model:** GLiNER2 Multi-v1 chưa train, và GLiNER2-PII (`fastino/gliner2-privacy-filter-PII-multi`). Cả hai đều chạy qua `evaluate.py`.
- **Corpus:** 9 corpus có validation (cả TAB). MAPA bị loại vì không có validation.
- **Giai đoạn 1, validation:** 3 cách × 9 corpus × 2 model, `--round val`, decode ở 0.5.
- **Luật chọn, cố định trước:**
  - Với mỗi cặp (model, corpus), chọn cách có rare-half macro-F1 cao nhất trên validation; hòa thì giữ S0.
  - Luật toàn cục: với mỗi model, cách có trung vị (validation − S0) cao nhất trên 9 corpus.
- **Giai đoạn 2, test:** chỉ chạy những (model, corpus, cách) được chọn mà chưa có dump test. Dump test S0 đã có từ 9.7/9.10, dump test S2 của model chưa train đã có từ 9.18.
- **Đại lượng chính:** test rare-half macro-F1 của cách được chọn trừ S0, theo từng (model, corpus). Báo cáo trung vị, số corpus tăng, và sign-flip trên 9 corpus cho mỗi model. Micro-F1 là phụ.
- **Luật đọc:**
  - Nếu cách chọn trên validation tăng điểm trên ≥ 6/9 corpus cho cả hai model, khuyến nghị: "khi dùng zero-shot, chọn tên nhãn trên validation".
  - Nếu không, khuyến nghị giữ tên gốc, và báo cáo là cách đặt tên không ổn định.
  - Tên cụ thể của cách thắng chỉ được báo cáo mang tính mô tả.
- **Kết quả (2026-10-01, `scripts/pii_naming.py --report`):**
  - **Dữ liệu:** 54 dump validation và 14 dump test mới, mỗi dump tải hai lần và hash khớp. Gretel-Finance có 1 mẫu OOM (văn bản 2663) ở mọi dump test, bị loại nhất quán.
  - **Validation:** cách được chọn hơn S0 ở cả 18/18 cặp. S2 thắng 6/9 corpus với model chưa train và 7/9 với GLiNER2-PII. S1 thắng ở Gretel, Nemotron và Gretel-Finance (cả hai model), riêng GLiNER2-PII thì chỉ Nemotron và Gretel-Finance. Luật toàn cục chọn S2 cho cả hai model (trung vị +4.69 / +4.81).
  - **Test, model chưa train:** tăng trên **8/9**, trung vị **+5.03**, p = 0.0078. Chỉ OpenPII giảm (−1.23).
  - **Test, GLiNER2-PII:** tăng trên **8/9**, trung vị **+6.17**, p = 0.031. Chỉ OpenPII giảm (−4.36).
  - **Luật toàn cục (S2 cho mọi corpus):** tăng trên 6/9 (+3.06) và 7/9 (+2.68).
  - **Micro-F1 (phụ):** tăng trên 6/9 với cả hai model.
  - ⇒ **Luật đọc đạt.** Khuyến nghị: khi dùng một model chưa train trên corpus đích, chọn tên nhãn trên validation.

## 10. Nhật ký sai lệch

| Ngày | Thay đổi | Lý do |
|---|---|---|
| 2026-09-27 (trước khi duyệt) | Arm (b) thêm `--full-grid` | Để (c) − (b) chỉ còn khác 3 term (mục 2) |
| 2026-09-27, khoảng 15:30, trước khi có kết quả seed 2024 nào | **Đổi hướng paper:** bỏ (c) làm phương pháp. Paper thành audit benchmark PII + độ tin cậy khi đánh giá loại hiếm + khuyến nghị khối (b). **Dừng** 3 run (c) seed 2024 (Gretel, OpenPII, Kiji) và dùng tiền tiết kiệm được để chạy **Privy (b) seed 2024**. Như vậy (b) có seed 42/1337/2024 trên cả 4 corpus; (a) và (c) giữ 2 seed. | Ngân sách không đủ cho cả hai. (b) là công thức được khuyến nghị, nên cần nhiều seed hơn; phần (c) chỉ còn là kết quả phụ ("các term không cần thiết"). |
| 2026-09-27, sau kết quả đợt 1, **trước khi chạy seed 2024** | **Đợt 2:** thêm seed 2024 cho (b) và (c) trên cả 4 corpus, đúng cấu hình của đợt 1. Seed 2024 chỉ được đưa vào phân tích khi có đủ cả 4 corpus cho cả (b) lẫn (c); `rare_pii_tables.py` tự áp luật này, nên một corpus còn thiếu sẽ giữ phân tích ở 2 seed. **Thêm 2 kiểm định**: (i) *sign-flip permutation* chính xác trên các hiệu (c)−(b) theo từng cặp (corpus, seed) ghép theo seed, rare-half ở threshold tuned; đơn vị là một lần train, nên kiểm định này tính cả dao động giữa các seed; (ii) *Wilcoxon signed-rank* trên hiệu F1 của từng nhãn hiếm (lấy trung bình theo seed), để xem mức tăng trải đều hay do vài nhãn kéo. Kết luận đợt 1 (bootstrap, kết cục B) giữ nguyên và được báo cáo trong phụ lục protocol của paper. (i) là kiểm định quyết định với luật "Δ > 0 và p hai phía < 0.05". (ii) và CI của bootstrap theo câu test đều được báo cáo kèm trong Tab. 6; ban đầu nhóm tác giả muốn bỏ bootstrap, sau đó yêu cầu giữ lại để đủ ba góc nhìn. Hai kiểm định này áp dụng cho mọi số seed đang có, kể cả 2 seed hiện tại. | Nhóm tác giả yêu cầu thêm seed và một kiểm định khác. Bootstrap theo câu test không tính dao động giữa các seed, trong khi trên Gretel hai seed của (c) lệch nhau 3 điểm. Ngân sách chưa đủ cho cả 4 corpus: Privy (khoảng $9.4) chờ thêm ngân sách. Chọn để lại Privy chứ không phải Kiji, vì Kiji là corpus duy nhất có (c)−(b) âm. |
| 2026-09-27, trước khi có kết quả nào của (b) | **Đợt 1 theo ngân sách ($54 trên 2 workspace).** D1 = dùng lại pilot: (a) và (c) lấy run v1 seed 42/1337; chỉ chạy (b) trên 4 corpus × seed 42/1337, đúng cấu hình GPU/batch/dtype của pilot, kết quả ghi vào `v1-bce-bundle-50k`. Hệ quả: 2 seed thay vì 5; threshold tuned chỉ trong lưới {0.50, 0.55, …, 0.90}; (b) không dump 0.1, giữ checkpoint seed 42. Endpoint chính và luật quyết định giữ nguyên. | Protocol đủ cần khoảng $200 (ước từ `elapsed_seconds` của pilot và giá Modal); đợt này khoảng $24–33. Seed 2024/7/13 và dump 0.1 để đợt sau nếu có thêm ngân sách. |
