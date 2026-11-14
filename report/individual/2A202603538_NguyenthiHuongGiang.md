# Báo cáo Day 7 – Retrieval Context và Agent

## 1. Dữ liệu và benchmark

- **Nguồn dữ liệu:** OpenStax Calculus Volume 1.
- **Quyền sử dụng:** CC BY-NC-SA 4.0 (theo manifest).
- **Phạm vi:** Trang 16–45, gồm 30 trang.
- **Số chunks:** 119, sử dụng `data/day6_chunks_A.json`.
- **Vector store:** PostgreSQL + pgvector.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Embedding dimension:** 384.
- **Agent model:** `openai/gpt-oss-20b`, thông qua Groq.
- **Benchmark:** 24 câu hỏi, gồm 17 câu Dev và 7 câu Test. Trong đó Dev có 15 câu answerable (D01–D15) và 2 câu unanswerable (D16–D17); Test có 5 câu answerable (T01–T05) và 2 câu unanswerable (T06–T07).
- **Đánh giá retrieval:** Chỉ tính các câu answerable. Do đó, mẫu số của Dev là 15 và Test là 5. Các câu unanswerable không được tính vào Hit@k.
- **Chỉ số:** Hit@1 đo tỷ lệ câu có evidence đúng ở vị trí đầu tiên; Hit@3 đo tỷ lệ câu có evidence trong ba kết quả đầu; Hit@5 đo tỷ lệ câu có evidence trong năm kết quả đầu. Context Evidence đo tỷ lệ câu mà evidence đúng được đưa vào context cuối cùng.

## 2. Retrieval và thử nghiệm

Các phương pháp được so sánh trên cùng tập câu hỏi Dev answerable. Context budget của các baseline là 1800 ký tự.

### 2.1. Kết quả baseline trên Dev

| Cấu hình | Phương pháp | Hit@1 | Hit@3 | Context Evidence | Median latency | Index time |
|---|---|---:|---:|---:|---:|---:|
| Baseline 1 | TF-IDF | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.73 ms | 0.036 s |
| Baseline 2 | BM25 | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.29 ms | 0.026 s |
| Baseline 3 | Dense, Top-3, budget 1800 | 6/15 (40.00%) | 12/15 (80.00%) | 12/15 (80.00%) | 23.93 ms | 3.045 s |

Chi phí dịch vụ retrieval ghi nhận ở các lần chạy: **$0**.

### 2.2. Thử nghiệm cải thiện Dense trên Dev

| Cấu hình | Thay đổi | Hit@1 | Hit@3 | Hit@5 | Context Evidence | Median latency |
|---|---|---:|---:|---:|---:|---:|
| Dense baseline | Top-3, budget 1800 | 6/15 | 12/15 | — | 12/15 | 23.93 ms |
| Cải thiện 1 | Top-5, budget 1800 | 6/15 | 12/15 | 14/15 | 12/15 | 21.43 ms |
| Cải thiện 2 | Top-3, budget 2400 | 6/15 | 12/15 | — | 12/15 | 23.01 ms |

Lưu ý: Kết quả Top-5 được tính trực tiếp từ `traces.json` bằng cách kiểm tra riêng ba và năm kết quả đầu. Trường `hit3` trong summary của cấu hình Top-5 thực tế đang đếm trên toàn bộ năm kết quả, vì vậy không sử dụng trường này làm Hit@3.

**Nhận xét:**

- Tăng `top_k` từ 3 lên 5 giúp Dense tăng khả năng tìm thấy evidence trong phạm vi tìm kiếm: Hit@5 đạt 14/15.
- Hit@3 vẫn là 12/15 và Context Evidence vẫn là 12/15. Như vậy, hai kết quả bổ sung chưa giúp tăng số câu có evidence đúng trong context cuối cùng.
- Tăng context budget từ 1800 lên 2400 với Top-3 không làm tăng `avg_context_chars` (vẫn khoảng 1713.33 ký tự) và không cải thiện các chỉ số retrieval.
- Trong phạm vi thử nghiệm, Dense Top-5 là cấu hình có recall ở Top-5 tốt nhất trên Dev. Tuy nhiên, lợi ích này chưa chuyển thành mức tăng Context Evidence.

### 2.3. Kết quả trên Test

Test gồm 5 câu answerable. Kết quả được dùng để kiểm tra cấu hình đã lựa chọn trên Dev, không dùng để lựa chọn lại cấu hình.

| Phương pháp | Hit@1 | Hit@3 | Context Evidence | Median latency |
|---|---:|---:|---:|---:|
| TF-IDF | 0/5 | 4/5 | Chưa có số liệu trong summary | 0.427 ms |
| BM25 | 1/5 | 5/5 | 5/5 | 0.469 ms |
| Dense Top-3, budget 1800 | 2/5 | 4/5 | 4/5 | 21.63 ms |
| Dense Top-5, budget 1800 | 2/5 | 5/5* | 4/5 | 20.34 ms |

\* Với Dense Top-5, giá trị `hit3 = 5` trong summary thực chất được tính trên cả năm kết quả. Vì vậy, đây là Hit@5, không phải Hit@3.

**Nhận xét:**

- BM25 đạt Context Evidence 5/5 trên Test và có latency thấp.
- Dense Top-5 đạt Hit@5 5/5, nhưng Context Evidence vẫn là 4/5.
- Dense Top-5 được giữ làm cấu hình cải thiện đã chọn theo kết quả Dev; kết quả Test được báo cáo riêng để kiểm tra khả năng tổng quát hóa.
- Kết quả Test chỉ gồm năm câu answerable nên chưa đủ để đưa ra kết luận tổng quát về hiệu năng trên các dạng câu hỏi khác.

### 2.4. Phân tích lỗi và hướng cải thiện

Một số lỗi retrieval được kiểm tra trực tiếp từ trace của Dense baseline:

1. **D03 – “What three components does a function consist of?”**: Evidence định nghĩa function nằm ở trang 16 nhưng không xuất hiện trong Top-3. Các kết quả được xếp hạng cao lại nói về function composition. Có khả năng embedding bị thu hút bởi từ khóa “function” và nội dung composition thay vì định nghĩa trực tiếp.
2. **D05 – “What is the range of a function?”**: Evidence định nghĩa range nằm ở trang 16 nhưng không được lấy vào Top-5. Một số chunk khác có nhắc đến range và domain nhưng không chứa đúng định nghĩa cần tìm.
3. **D06 – “Can two different inputs of a function produce the same output?”**: Evidence trực tiếp nằm ở trang 17 nhưng không xuất hiện trong Top-3. Các kết quả gần nhất nói về input/output hoặc composition nhưng chưa thể hiện đúng quan hệ được hỏi.

Thử nghiệm tăng Top-k giúp giảm số câu bị miss ở phạm vi tìm kiếm, nhưng chưa cải thiện Context Evidence do giới hạn context budget và cách chọn chunk. Hướng tiếp theo là cải thiện query formulation, reranking và chiến lược chọn context, đồng thời kiểm tra khả năng giữ lại chunk chứa định nghĩa trực tiếp.

## 3. Agent và đánh giá

### 3.1. Cơ chế hoạt động

- Agent nhận câu hỏi, sử dụng retrieval tool để tìm các chunks liên quan và đưa kết quả vào context cho LLM.
- LLM tạo câu trả lời dựa trên context và cung cấp citation tham chiếu đến chunks.
- Hệ thống có kiểm tra citation ID có tồn tại trong tập chunks hay không.
- Nếu citation ID hợp lệ nhưng chưa xuất hiện trong phần câu trả lời, hệ thống có thể bổ sung ID tham chiếu.
- Cơ chế hiện tại chủ yếu xác minh tính hợp lệ của citation ID; chưa tự động xác minh đầy đủ rằng nội dung chunk thực sự hỗ trợ kết luận.
- Trace được lưu để kiểm tra query, kết quả retrieval, context và citation.

### 3.2. Kết quả chạy Agent

| Split | Số câu trong lần chạy | Có câu trả lời | Lỗi | Nhận xét |
|---|---:|---:|---:|---|
| Dev | 15 | 14 | 1 | D03 gặp lỗi provider/tool calling HTTP 400 |
| Test | 5 | 5 | 0 | Có phản hồi cho cả năm câu |

Ở Dev, D03 gặp lỗi HTTP 400 với thông báo `tool_use_failed` và `Tool choice is none, but model called a tool`. Đây là lỗi tương tác giữa model và tool của provider, không thể xem là bằng chứng trực tiếp cho thấy retrieval thất bại.

Các câu Test đã có phản hồi gồm:

- **T01:** Trả lời về tỷ số thay đổi Δy/Δx; citation p39, fixed 1 và 2.
- **T02:** Trả lời về đường nằm ngang khi độ dốc bằng 0; citation p39, fixed 1.
- **T03:** Trả lời về vertical line test; citation p22, fixed 5.
- **T04:** Trả lời về định nghĩa horizontal asymptote; citation p43, fixed 1.
- **T05:** Trả lời về leading term và end behavior; citation p43, fixed 3 và 2.

Các citation trên cần được đối chiếu với nội dung chunk tương ứng trước khi kết luận chúng hỗ trợ chính xác câu trả lời. Việc Agent có phản hồi không đồng nghĩa với câu trả lời đã được xác nhận grounded.

### 3.3. Trace và giới hạn đánh giá

- Dev trace: `outputs/agent_traces_dev.json`.
- Test trace: `outputs/agent_traces_test.json`.
- Retrieval traces và summary được lưu trong các thư mục tương ứng dưới `outputs/`.

Các số liệu answer grade 0/1/2, semantic citation support, citation coverage, abstention đúng và false abstention chưa được chấm thủ công đầy đủ. Vì vậy, báo cáo không xem tỷ lệ có phản hồi là tỷ lệ grounded answer đúng và không tự gán điểm cho các câu chưa được kiểm tra.

Bộ benchmark đã có các câu unanswerable D16, D17, T06 và T07. Tuy nhiên, các kết quả Agent được tổng hợp ở trên là các lần chạy trên 15 câu Dev và 5 câu Test answerable; chưa có kết quả riêng đủ tin cậy để đánh giá khả năng abstention trên nhóm unanswerable.

Chi phí retrieval ghi nhận: $0. Chi phí LLM và tổng thời gian chạy Agent chưa có số liệu tổng hợp đáng tin cậy.

## 4. Thử thách nâng cao

Chưa triển khai đánh giá riêng cho bảng, ảnh, sơ đồ, biểu đồ hoặc multi-hop. Phạm vi hiện tại tập trung vào chunks văn bản trong OpenStax Calculus Volume 1.

| Nhóm test | Trạng thái | Nhận xét |
|---|---|---|
| Text retrieval | Đã thực hiện | So sánh TF-IDF, BM25 và Dense |
| Bảng | Chưa đánh giá | Chưa có benchmark riêng |
| Ảnh, sơ đồ | Chưa đánh giá | Chưa triển khai |
| Biểu đồ | Chưa đánh giá | Chưa triển khai |
| Multi-hop | Chưa đánh giá | Chưa có benchmark riêng |
| Thiếu bằng chứng | Có câu hỏi trong benchmark | Chưa có đánh giá Agent abstention đầy đủ |

## 5. Tái lập và giới hạn

- Dữ liệu chunks: `data/day6_chunks_A.json`.
- Bộ câu hỏi: `data/questions_openstax_en.json`.
- Các script retrieval nằm trong `starter/`.
- Kết quả và trace nằm trong `outputs/`.
- Dense sử dụng môi trường Python trong WSL và model `sentence-transformers/all-MiniLM-L6-v2`.
- TF-IDF và BM25 chạy trực tiếp bằng Python trong PowerShell.
- Các lần chạy được ghi nhận có chi phí dịch vụ retrieval bằng $0; không có số liệu tổng hợp đáng tin cậy về chi phí LLM.

Các giới hạn chính:
1. Corpus chỉ gồm 30 trang của một tài liệu, nên chưa đại diện cho nhiều loại tài liệu và lĩnh vực.
2. Dense có latency và thời gian index cao hơn đáng kể so với TF-IDF và BM25 trong môi trường thử nghiệm.
3. Tăng Top-k cải thiện khả năng tìm thấy evidence nhưng chưa chắc làm tăng Context Evidence.
4. Citation ID validation chưa thay thế được đánh giá semantic citation support.
5. Agent chưa được chấm thủ công đầy đủ về grounded answer và khả năng abstention.
6. Các kết quả Test có số lượng câu answerable nhỏ, nên cần thận trọng khi diễn giải.

## 6. Kết luận

Trong Day 07, tôi đã triển khai và so sánh ba phương pháp retrieval gồm TF-IDF, BM25 và Dense Retrieval trên cùng benchmark. Kết quả Dev cho thấy Dense đạt Context Evidence cao hơn hai phương pháp sparse (12/15 so với 10/15), trong khi BM25 có latency thấp hơn đáng kể. Thử nghiệm Dense Top-5 tăng Hit@5 lên 14/15 trên Dev, nhưng Context Evidence không tăng. Thử nghiệm tăng budget lên 2400 cũng chưa đem lại cải thiện.

Trên Test, BM25 đạt Context Evidence 5/5; Dense Top-5 đạt Hit@5 5/5 nhưng Context Evidence 4/5. Cấu hình Dense Top-5 được giữ làm cấu hình cải thiện theo kết quả Dev, không lựa chọn lại dựa trên Test.

Qua quá trình thực hiện, tôi nhận thấy chất lượng RAG không thể chỉ đánh giá bằng việc hệ thống có tạo được câu trả lời hay không. Cần đánh giá riêng khả năng truy xuất evidence, khả năng đưa evidence vào context, mức độ grounded của câu trả lời, tính chính xác về ngữ nghĩa của citation và khả năng từ chối khi tài liệu không có thông tin. Đây cũng là những phần cần tiếp tục hoàn thiện để hệ thống có thể được đánh giá đáng tin cậy hơn.
