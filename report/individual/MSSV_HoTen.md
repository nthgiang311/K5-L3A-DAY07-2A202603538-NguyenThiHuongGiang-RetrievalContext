# Báo cáo Day 7 – Retrieval Context và Agent

## Dữ liệu và benchmark

- **Nguồn:** OpenStax Calculus Volume 1.
- **Quyền sử dụng:** CC BY-NC-SA 4.0 (theo thông tin manifest của dữ liệu).
- **Phạm vi dữ liệu:** Trang 16–45, gồm 30 trang.
- **Số chunk:** 119 chunks, sử dụng file `data/day6_chunks_A.json`.
- **Vector store:** PostgreSQL + pgvector.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Số chiều embedding:** 384.
- **Agent model:** `openai/gpt-oss-20b`, sử dụng thông qua Groq.
- **Cách chạy:** Chạy các script Python trong thư mục `starter/`, sử dụng dữ liệu chunks và bộ câu hỏi JSON.
- **Phân chia benchmark:** 15 câu Dev (D01–D15), 5 câu Test (T01–T05). Bộ câu hỏi hiện tại chưa có các trường phân loại câu hỏi thiếu evidence.
- **Cách xác định câu có/thiếu evidence:** Chưa hoàn thiện quy trình phân loại answerable/unanswerable. Các câu hỏi hiện có được xây dựng xoay quanh nội dung tài liệu; chưa có kết quả đánh giá chính thức cho nhóm câu hỏi thiếu evidence.
- **Tiêu chí chấm retrieval:** Hit@1, Hit@3 và Context Hit. Hit@1 kiểm tra evidence đúng có xuất hiện ở vị trí đầu tiên hay không; Hit@3 kiểm tra evidence có nằm trong ba kết quả đầu hay không; Context Hit kiểm tra evidence có xuất hiện trong context được đưa vào hệ thống hay không.

## Retrieval và thử nghiệm

Context budget sử dụng trong các lần chạy so sánh là 1800 ký tự. Các phương pháp được đánh giá trên cùng tập 15 câu hỏi Dev.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF; context budget 1800 ký tự | Mốc so sánh sử dụng sparse retrieval | dev | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | Index: 0.015 s; median: 0.539 ms; $0 |
| Phương pháp 2 | BM25; context budget 1800 ký tự | Thay TF-IDF bằng BM25 để cải thiện xếp hạng theo mức độ liên quan từ khóa | dev | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | Index: 0.0076 s; median: 0.367 ms; $0 |
| Phương pháp 3 | Dense Retrieval; embedding 384 chiều; context budget 1800 ký tự | Sử dụng semantic embedding thay vì chỉ dựa trên khớp từ khóa | dev | 6/15 (40.00%) | 12/15 (80.00%) | 12/15 (80.00%) | Index: 2.710 s; median: 57.762 ms; $0 |
| Cải thiện 1 | Chưa có kết quả chạy riêng | Chưa xác định | dev | Chưa có | Chưa có | Chưa có | Chưa có |
| Cải thiện 2 | Chưa có kết quả chạy riêng | Chưa xác định | dev | Chưa có | Chưa có | Chưa có | Chưa có |
| Cấu hình đã chọn | Chưa chốt cấu hình cuối cùng | Cần đánh giá và lựa chọn dựa trên kết quả Dev | test | Chưa có | Chưa có | Chưa có | Chưa có |
| Baseline | Chưa có kết quả retrieval Test riêng | Mốc đối chiếu trên Test cần được chạy | test | Chưa có | Chưa có | Chưa có | Chưa có |

**Giải thích kết quả và lựa chọn cấu hình**

- TF-IDF và BM25 có cùng kết quả Hit@1, Hit@3 và Context Hit trên tập Dev.
- BM25 có median latency thấp nhất trong ba phương pháp.
- Dense Retrieval đạt Evidence Hit@3 và Context Evidence Rate cao nhất, cùng đạt 12/15 câu. Tuy nhiên, Hit@1 chỉ đạt 6/15 và median latency cao hơn đáng kể.
- Kết quả cho thấy Dense có lợi thế về khả năng đưa evidence vào nhóm Top-3, trong khi BM25 có lợi thế về tốc độ trong thử nghiệm hiện tại.
- Chưa chốt cấu hình cuối cùng vì cần thực hiện thêm hai thử nghiệm cải thiện và đánh giá trên Test sau khi lựa chọn cấu hình dựa trên Dev.

**Hai lỗi Dev và hướng xử lý**

1. **Lỗi truy xuất không đưa evidence đúng vào vị trí đầu:** Dense có Hit@1 thấp hơn TF-IDF và BM25 (6/15 so với 8/15). Tuy nhiên, Dense có Hit@3 cao hơn. Điều này cho thấy cần phân biệt lỗi xếp hạng ở vị trí đầu với lỗi không tìm thấy evidence trong Top-3. Hướng xử lý tiếp theo là kiểm tra các query bị miss và thử nghiệm điều chỉnh chiến lược retrieval.
2. **Lỗi gọi tool từ LLM ở câu D03:** Provider trả về HTTP 400 với thông báo `tool_use_failed` và `Tool choice is none, but model called a tool`. Đây là lỗi tương tác với provider/tool, không phải bằng chứng trực tiếp cho thấy retrieval thất bại. Cần kiểm tra cấu hình tool calling và cách model lựa chọn tool; không nên tính trường hợp này là một câu trả lời đúng hoặc tự tạo kết quả thay thế.

## Agent và đánh giá

**Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi**

- Agent sử dụng kết quả retrieval làm context để LLM tạo câu trả lời.
- Các chunks được chọn từ kết quả truy xuất và đưa vào context của LLM.
- Hệ thống có kiểm tra citation ID có tồn tại trong tập chunks hay không.
- Nếu citation ID hợp lệ nhưng chưa được đưa vào câu trả lời, hệ thống có thể bổ sung ID tham chiếu vào nội dung trả lời.
- Citation validation hiện tại kiểm tra tính hợp lệ của ID, chưa xác minh đầy đủ rằng nội dung chunk thực sự hỗ trợ kết luận.
- Số lượt gọi tool cụ thể, điều kiện dừng và thống kê từng loại tool cần được trích xuất từ trace trước khi điền số liệu.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Agent / Dev | Có sử dụng tool trong pipeline | Chưa tổng hợp | Chưa chấm thủ công | Có kiểm tra ID | Chưa thống kê đầy đủ | Chưa đánh giá | Chưa đánh giá |
| Agent / Test | Có sử dụng tool trong pipeline | Chưa tổng hợp | Chưa chấm thủ công | Có kiểm tra ID | Chưa thống kê đầy đủ | Chưa đánh giá | Chưa đánh giá |

### Kết quả chạy Agent

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| D01–D02 | Dev | Có phản hồi | Có câu trả lời | Chưa chấm grounded answer thủ công |
| D03 | Dev | Lỗi HTTP 400 `tool_use_failed` | Không có kết quả hoàn chỉnh đáng tin cậy để chấm | Lỗi provider/tool calling |
| D04–D15 | Dev | Có phản hồi sau khi xử lý các lần chạy | Có câu trả lời | Chưa chấm grounded answer thủ công |
| T01 | Test – độ dốc | Có phản hồi | Trả lời về tỷ số thay đổi Δy/Δx; citation p39, fixed 1 và 2 | Cần đối chiếu lại nội dung nguồn |
| T02 | Test – độ dốc bằng 0 | Có phản hồi | Trả lời về đường nằm ngang; citation p39, fixed 1 | Cần đối chiếu lại nội dung nguồn |
| T03 | Test – đường thẳng đứng | Có phản hồi | Trả lời về vertical line test; citation p22, fixed 5 | Cần đối chiếu lại nội dung nguồn |
| T04 | Test – tiệm cận ngang | Có phản hồi | Trả lời về định nghĩa horizontal asymptote; citation p43, fixed 1 | Cần đối chiếu lại nội dung nguồn |
| T05 | Test – hành vi đa thức | Có phản hồi | Trả lời về leading term và end behavior; citation p43, fixed 3 và 2 | Cần đối chiếu lại nội dung nguồn |

**Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí**

- Dev trace: `outputs/agent_traces_dev.json`.
- Test trace: `outputs/agent_traces_test.json`.
- Retrieval TF-IDF: `outputs/day7_dev_tfidf_1800/`.
- Retrieval BM25: `outputs/day7_dev_bm25_1800/`.
- Retrieval Dense: `outputs/day7_dev_dense_1800/`.
- Các thư mục retrieval có `summary.json` và `traces.json`.
- Lệnh chạy Dense đã sử dụng:

```powershell
wsl -d Ubuntu-D -- /mnt/d/K5-L3A-DAY06-NguyenThiHuonGiang-2A202603538-DataIngestion/starter/.venv/bin/python -u /mnt/d/K5-L3A-DAY07-2A202603538-NguyenThiHuongGiang-RetrievalContext/starter/evaluate_dense.py --chunks /mnt/d/K5-L3A-DAY07-2A202603538-NguyenThiHuongGiang-RetrievalContext/data/day6_chunks_A.json --questions /mnt/d/K5-L3A-DAY07-2A202603538-NguyenThiHuongGiang-RetrievalContext/data/questions_openstax_en.json --split dev --output /mnt/d/K5-L3A-DAY07-2A202603538-NguyenThiHuongGiang-RetrievalContext/outputs/day7_dev_dense
```

- Chi phí dịch vụ retrieval ghi nhận: $0.
- Chi phí gọi LLM và tổng thời gian chạy Agent: chưa có số liệu tổng hợp đáng tin cậy.

**Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn**

- So sánh ba phương pháp retrieval cho thấy BM25 có tốc độ tốt, trong khi Dense có Context Hit cao hơn.
- D03 là ví dụ về lỗi provider/tool calling, cần tách biệt với lỗi chất lượng retrieval.
- Agent đã trả lời được 14/15 câu Dev và 5/5 câu Test trong các lần chạy được ghi nhận. Tuy nhiên, tỷ lệ có phản hồi không đồng nghĩa với tỷ lệ grounded answer đúng.
- Citation ID được kiểm tra về tính hợp lệ nhưng chưa được xác minh đầy đủ về ngữ nghĩa.
- Tập Test mới có 5 câu và chưa có nhóm câu hỏi thiếu evidence; do đó chưa đánh giá được khả năng abstention và từ chối nhầm.

## Thử thách nâng cao (nếu thực hiện)

Chưa triển khai đánh giá nâng cao trên bảng, ảnh, sơ đồ và biểu đồ. Phần thử nghiệm hiện tại tập trung vào nội dung văn bản trong OpenStax Calculus Volume 1.

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** OpenStax Calculus Volume 1, CC BY-NC-SA 4.0, phạm vi trang 16–45.
- **Cách xử lý text, bảng, ảnh, biểu đồ:** Phạm vi hiện tại tập trung vào chunks văn bản. Chưa có benchmark riêng để đánh giá khả năng trích xuất bảng, ảnh, sơ đồ hoặc biểu đồ.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên Dev:** Chưa triển khai đầy đủ benchmark nâng cao.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | Chưa đánh giá | Chưa có | Chưa có benchmark riêng |
| Ảnh, sơ đồ | Chưa đánh giá | Chưa có | Chưa triển khai |
| Biểu đồ | Chưa đánh giá | Chưa có | Chưa triển khai |
| Multi-hop | Chưa đánh giá | Chưa có | Chưa có benchmark riêng |
| Text | Chưa đánh giá theo nhóm nâng cao | Chưa có | Đã có benchmark retrieval cơ bản |
| Thiếu bằng chứng | Chưa đánh giá | Chưa có | Chưa bổ sung câu hỏi unanswerable |
| **Toàn bộ** | Chưa đánh giá | Chưa có | Chưa triển khai thử thách nâng cao |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** Chưa có bộ trace và nhãn riêng cho thử thách nâng cao.
- **Giới hạn, cách tái lập và đường dẫn benchmark:** Có thể tái lập các thử nghiệm retrieval cơ bản bằng các script trong `starter/`, dữ liệu trong `data/` và kết quả trong `outputs/`. Thử thách nâng cao chưa có benchmark hoàn chỉnh.

---

## Kết luận

Trong Day 07, tôi đã triển khai pipeline retrieval với ba phương pháp TF-IDF, BM25 và Dense Retrieval, đồng thời kết nối retrieval với LLM để xây dựng Agentic RAG có citation. Kết quả ban đầu cho thấy sự đánh đổi giữa khả năng tìm evidence trong Top-k và tốc độ xử lý.

Qua quá trình thực hiện, tôi nhận thấy đánh giá một hệ thống RAG không thể chỉ dựa trên việc hệ thống có tạo được câu trả lời hay không. Cần đánh giá riêng khả năng truy xuất evidence, mức độ grounded của câu trả lời, tính chính xác của citation và khả năng từ chối khi tài liệu không có thông tin. Đây là những nội dung cần tiếp tục hoàn thiện để có thể đưa ra kết luận đáng tin cậy hơn về chất lượng hệ thống.
