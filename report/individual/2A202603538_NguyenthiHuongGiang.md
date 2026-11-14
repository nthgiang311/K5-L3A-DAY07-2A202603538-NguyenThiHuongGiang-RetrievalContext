# Báo cáo Day 7 – Retrieval Context và Agent

## 1. Dữ liệu và benchmark

### 1.1. Dữ liệu sử dụng

- **Nguồn dữ liệu:** OpenStax Calculus Volume 1.
- **Quyền sử dụng:** CC BY-NC-SA 4.0 (theo thông tin trong manifest).
- **Phạm vi dữ liệu:** Trang 16–45, gồm 30 trang.
- **Số lượng chunks:** 119 chunks, sử dụng file `data/day6_chunks_A.json`.
- **Vector store:** PostgreSQL + pgvector.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Embedding dimension:** 384.
- **Agent model:** `openai/gpt-oss-20b`, sử dụng thông qua Groq.

### 1.2. Bộ câu hỏi và cách đánh giá

Bộ benchmark gồm 24 câu hỏi, được chia thành hai tập:

| Split | Tổng số câu | Answerable | Unanswerable |
|---|---:|---:|---:|
| Dev | 17 | 15 (D01–D15) | 2 (D16–D17) |
| Test | 7 | 5 (T01–T05) | 2 (T06–T07) |
| **Tổng** | **24** | **20** | **4** |

Các chỉ số retrieval được tính trên nhóm câu hỏi answerable. Các câu unanswerable không được tính vào Hit@k vì không có evidence tương ứng để truy xuất.

Các chỉ số sử dụng:

- **Hit@1:** Tỷ lệ câu hỏi có evidence đúng nằm ở kết quả truy xuất đầu tiên.
- **Hit@3:** Tỷ lệ câu hỏi có evidence đúng nằm trong ba kết quả đầu tiên.
- **Hit@5:** Tỷ lệ câu hỏi có evidence đúng nằm trong năm kết quả đầu tiên.
- **Context Evidence:** Tỷ lệ câu hỏi mà evidence đúng thực sự được đưa vào context cuối cùng gửi cho LLM.
- **Median latency:** Thời gian truy xuất trung vị trên mỗi câu hỏi.
- **Index time:** Thời gian xây dựng index cho phương pháp tương ứng.

Việc đánh giá Agent được thực hiện riêng với retrieval. Một câu trả lời có phản hồi hoặc có citation ID hợp lệ chưa đủ để kết luận câu trả lời đã được grounded; cần kiểm tra xem nội dung evidence có thực sự hỗ trợ kết luận hay không.

## 2. Retrieval và thử nghiệm

### 2.1. Kết quả baseline trên Dev

Ba phương pháp TF-IDF, BM25 và Dense Retrieval được đánh giá trên cùng 15 câu hỏi answerable của tập Dev. Context budget sử dụng trong các lần chạy so sánh baseline là 1800 ký tự.

| Cấu hình | Phương pháp | Hit@1 | Hit@3 | Context Evidence | Median latency | Index time |
|---|---|---:|---:|---:|---:|---:|
| Baseline 1 | TF-IDF | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.73 ms | 0.036 s |
| Baseline 2 | BM25 | 8/15 (53.33%) | 10/15 (66.67%) | 10/15 (66.67%) | 1.29 ms | 0.026 s |
| Baseline 3 | Dense Top-3, budget 1800 | 6/15 (40.00%) | 12/15 (80.00%) | 12/15 (80.00%) | 23.93 ms | 3.045 s |

Chi phí dịch vụ retrieval ghi nhận trong các lần chạy là **$0**.

**Nhận xét:**

- TF-IDF và BM25 có kết quả Hit@1, Hit@3 và Context Evidence giống nhau trên tập Dev.
- BM25 có median latency thấp hơn TF-IDF trong lần thử nghiệm này.
- Dense Retrieval có Hit@1 thấp hơn hai phương pháp sparse, nhưng Hit@3 và Context Evidence cao hơn.
- Dense có khả năng đưa evidence đúng vào nhóm kết quả đầu tốt hơn trong thử nghiệm này, nhưng phải đánh đổi bằng thời gian truy xuất và thời gian index cao hơn đáng kể.

Như vậy, không có một phương pháp tốt nhất trên mọi tiêu chí. BM25 có lợi thế về tốc độ, trong khi Dense có lợi thế về khả năng tìm evidence trong Top-3 và đưa evidence vào context.

### 2.2. Thử nghiệm cải thiện Dense trên Dev

Sau khi có kết quả baseline, tôi thực hiện hai thử nghiệm để kiểm tra ảnh hưởng của `top_k` và context budget.

| Cấu hình | Thay đổi | Hit@1 | Hit@3 | Hit@5 | Context Evidence | Median latency |
|---|---|---:|---:|---:|---:|---:|
| Dense baseline | Top-3, budget 1800 | 6/15 | 12/15 | — | 12/15 | 23.93 ms |
| Cải thiện 1 | Top-5, budget 1800 | 6/15 | 12/15 | 14/15 | 12/15 | 21.43 ms |
| Cải thiện 2 | Top-3, budget 2400 | 6/15 | 12/15 | — | 12/15 | 23.01 ms |

**Lưu ý về cách tính:** Kết quả Dense Top-5 được kiểm tra trực tiếp từ `traces.json`, tính riêng số câu có evidence trong Top-3 và Top-5. Trường `hit3` trong summary của cấu hình Top-5 thực tế đang đếm trên cả năm kết quả. Vì vậy, không sử dụng giá trị đó làm Hit@3.

**Nhận xét và lựa chọn cấu hình:**

- Tăng `top_k` từ 3 lên 5 giúp Dense tìm thấy evidence trong phạm vi năm kết quả đầu ở 14/15 câu, tương ứng Hit@5 đạt 93.33%.
- Hit@3 vẫn là 12/15 và Context Evidence vẫn là 12/15. Điều này cho thấy hai kết quả bổ sung giúp tăng khả năng tìm thấy evidence trong danh sách truy xuất, nhưng chưa làm tăng số câu có evidence trong context cuối cùng.
- Tăng context budget từ 1800 lên 2400 với Top-3 không cải thiện các chỉ số retrieval. `avg_context_chars` vẫn khoảng 1713.33 ký tự trong kết quả ghi nhận.
- Trong phạm vi thử nghiệm trên Dev, Dense Top-5 có Hit@5 cao nhất. Tuy nhiên, lợi ích này chưa chuyển thành mức tăng Context Evidence.

Tôi chọn Dense Top-5, budget 1800 làm cấu hình cải thiện để tiếp tục báo cáo và kiểm tra trên Test. Việc lựa chọn dựa trên kết quả Dev, không dựa trên kết quả Test.

### 2.3. Kết quả trên Test

Tập Test có 5 câu hỏi answerable. Kết quả được sử dụng để kiểm tra cấu hình sau khi đã lựa chọn trên Dev.

| Phương pháp | Hit@1 | Hit@3 | Hit@5 | Context Evidence | Median latency |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0/5 (0%) | 4/5 (80%) | — | Chưa có số liệu trong summary | 0.427 ms |
| BM25 | 1/5 (20%) | 5/5 (100%) | — | 5/5 (100%) | 0.469 ms |
| Dense Top-3, budget 1800 | 2/5 (40%) | 4/5 (80%) | — | 4/5 (80%) | 21.627 ms |
| Dense Top-5, budget 1800 | 2/5 (40%) | Chưa tách được | 5/5 (100%) | 4/5 (80%) | 20.340 ms |

**Lưu ý:** Với Dense Top-5, trường `hit3 = 5` trong summary thực chất được tính trên cả năm kết quả. Do đó, kết quả được ghi nhận là Hit@5 = 5/5, không phải Hit@3.

**Nhận xét:**

- BM25 đạt Context Evidence 5/5 trên Test và có latency thấp nhất trong các phương pháp được so sánh.
- Dense Top-5 đạt Hit@5 5/5 nhưng Context Evidence vẫn là 4/5.
- Dense Top-5 được giữ làm cấu hình cải thiện đã lựa chọn trên Dev. Kết quả Test được báo cáo riêng để kiểm tra khả năng tổng quát hóa, không dùng để lựa chọn lại cấu hình.
- Test chỉ có năm câu answerable nên cần thận trọng khi diễn giải kết quả. Chưa thể kết luận phương pháp nào sẽ tốt hơn trên các tài liệu hoặc dạng câu hỏi khác.

### 2.4. Phân tích lỗi retrieval và hướng cải thiện

Một số trường hợp chưa truy xuất được evidence trực tiếp đã được kiểm tra từ trace của Dense baseline.

1. **D03 – “What three components does a function consist of?”**

   Evidence định nghĩa function nằm ở trang 16 nhưng không xuất hiện trong Top-3. Các kết quả được xếp hạng cao lại đề cập đến function composition. Có khả năng embedding bị thu hút bởi từ khóa “function” và nội dung composition thay vì định nghĩa trực tiếp.

2. **D05 – “What is the range of a function?”**

   Evidence định nghĩa range nằm ở trang 16 nhưng không được lấy vào Top-5. Một số chunk khác có nhắc đến range và domain nhưng không chứa đúng định nghĩa cần tìm.

3. **D06 – “Can two different inputs of a function produce the same output?”**

   Evidence trực tiếp nằm ở trang 17 nhưng không xuất hiện trong Top-3. Các kết quả gần nhất đề cập đến input/output hoặc composition nhưng chưa thể hiện đúng quan hệ được hỏi.

Thử nghiệm tăng Top-k giúp giảm số câu bị miss trong danh sách truy xuất, nhưng chưa cải thiện Context Evidence. Điều này cho thấy cần phân biệt giữa việc tìm thấy evidence trong danh sách kết quả và việc thực sự đưa evidence đó vào context cuối cùng.

Các hướng cải thiện tiếp theo:

- Điều chỉnh query formulation để câu hỏi thể hiện rõ khái niệm cần tìm.
- Thử nghiệm reranking nhằm đưa chunk có nội dung hỗ trợ trực tiếp lên thứ hạng cao hơn.
- Cải thiện chiến lược chọn context, tránh để các chunk liên quan chung chung chiếm vị trí của evidence trực tiếp.
- Kiểm tra riêng các trường hợp định nghĩa, tính chất và ví dụ để đánh giá khả năng truy xuất theo từng dạng nội dung.

## 3. Agent và đánh giá

### 3.1. Cơ chế hoạt động

Agent được xây dựng bằng cách kết hợp retrieval tool với LLM để trả lời câu hỏi dựa trên nội dung tài liệu.

Quy trình hoạt động tổng quát:

1. Agent nhận câu hỏi từ người dùng.
2. Agent sử dụng retrieval tool để tìm các chunks liên quan trong dữ liệu đã nạp.
3. Các chunks được truy xuất được đưa vào context cho LLM.
4. LLM tạo câu trả lời dựa trên context và cung cấp citation tham chiếu đến chunks.
5. Hệ thống kiểm tra citation ID có tồn tại trong tập chunks hay không.
6. Trace được lưu để kiểm tra câu hỏi, kết quả retrieval, context và câu trả lời.

Hệ thống có cơ chế kiểm tra tính hợp lệ của citation ID. Nếu citation ID hợp lệ nhưng chưa xuất hiện trong phần câu trả lời, hệ thống có thể bổ sung ID tham chiếu.

Tuy nhiên, cơ chế này chủ yếu xác minh citation ID có tồn tại hay không, chưa tự động xác minh đầy đủ rằng nội dung chunk thực sự hỗ trợ kết luận. Vì vậy, cần phân biệt giữa citation hợp lệ về mặt định danh và citation có hỗ trợ đúng về mặt ngữ nghĩa.

### 3.2. Kết quả chạy Agent

Trace của Agent được ghi nhận trên 15 câu Dev answerable và 5 câu Test answerable.

| Split | Số câu trong lần chạy | Có câu trả lời | Lỗi |
|---|---:|---:|---:|
| Dev | 15 | 14 | 1 |
| Test | 5 | 5 | 0 |

Ở Dev, D03 gặp lỗi HTTP 400 với thông báo:

`tool_use_failed` – `Tool choice is none, but model called a tool`.

Đây là lỗi tương tác giữa model và tool của provider Groq. Trường hợp này không thể xem là bằng chứng trực tiếp cho thấy retrieval thất bại, đồng thời cũng không có câu trả lời hoàn chỉnh để đánh giá grounded answer.

### 3.3. Đánh giá thủ công câu trả lời và citation

Tôi kiểm tra nội dung câu trả lời và đối chiếu với context trong trace để đánh giá sơ bộ mức độ grounded.

Quy ước chấm:

- **Điểm 2:** Câu trả lời đúng và evidence trong context hỗ trợ nội dung được trả lời.
- **Điểm 1:** Ý chính tương đối đúng nhưng citation hoặc một phần giải thích chưa có đủ căn cứ trực tiếp.
- **Điểm 0:** Không có câu trả lời hoàn chỉnh để đánh giá hoặc câu trả lời không đáp ứng yêu cầu.

Kết quả đánh giá thủ công:

| Split | Số câu | Điểm 2 | Điểm 1 | Điểm 0 / lỗi | Điểm trung bình |
|---|---:|---:|---:|---:|---:|
| Dev | 15 | 12 | 2 | 1 | 1.73/2 |
| Test | 5 | 5 | 0 | 0 | 2.00/2 |

Điểm trung bình Dev được tính trên toàn bộ 15 trường hợp, trong đó D03 được tính 0 do lỗi và không có câu trả lời hoàn chỉnh. Đây là kết quả rà soát thủ công trên các trace hiện có, không phải đánh giá tự động toàn diện.

Một số trường hợp đáng chú ý:

- **D03:** Lỗi provider/tool calling HTTP 400, không có câu trả lời hoàn chỉnh để chấm.
- **D06:** Câu trả lời đúng về việc hai input khác nhau có thể tạo ra cùng output, nhưng citation được sử dụng chưa trực tiếp chứng minh tính chất này. Trường hợp được chấm 1.
- **D08:** Ý chính về các cách biểu diễn hàm có căn cứ, nhưng câu trả lời có thêm phần giải thích chưa được evidence hỗ trợ đầy đủ. Trường hợp được chấm 1.
- **Các câu Dev còn lại đã trả lời:** Được chấm 2 trong lần rà soát thủ công.
- **T01–T05:** Cả năm câu trả lời đều được chấm 2 trong lần rà soát thủ công.

Kết quả kiểm tra citation ID trong context:

| Split | Số trace có câu trả lời | Citation ID nằm trong context |
|---|---:|---:|
| Dev | 14 | 14/14 |
| Test | 5 | 5/5 |

Kết quả trên cho thấy citation ID trong các câu trả lời đã kiểm tra đều tham chiếu đến ID có trong context. Tuy nhiên, điều này không đồng nghĩa với việc mọi citation đều hỗ trợ đầy đủ nội dung câu trả lời. D06 và D08 là hai ví dụ cho thấy vẫn cần kiểm tra semantic citation support.

### 3.4. Trace, điều kiện đánh giá và giới hạn

Các file trace sử dụng:

- Dev trace: `outputs/agent_traces_dev.json`.
- Test trace: `outputs/agent_traces_test.json`.

Trong các trace đã kiểm tra, mỗi câu trả lời thành công thực hiện một lượt tìm kiếm. D03 là trường hợp lỗi provider/tool calling.

Các giới hạn đánh giá:

- Trace hiện tại chỉ bao gồm 15 câu Dev answerable và 5 câu Test answerable.
- Chưa có kết quả Agent riêng đủ tin cậy cho D16, D17, T06 và T07.
- Vì vậy, chưa thể đánh giá khả năng abstention của Agent khi tài liệu không chứa thông tin cần thiết.
- Chưa có số liệu đầy đủ về tỷ lệ từ chối đúng và từ chối nhầm.
- Việc kiểm tra citation ID mới xác minh tính hợp lệ của ID, chưa thay thế được việc đánh giá nội dung citation có hỗ trợ kết luận hay không.
- Điểm grounded answer được đánh giá thủ công trên các trace hiện có, chưa phải một quy trình chấm tự động hoặc đánh giá độc lập bởi nhiều người.

Chi phí dịch vụ retrieval ghi nhận là $0. Chi phí gọi LLM và tổng thời gian chạy Agent chưa có số liệu tổng hợp đáng tin cậy.

## 4. Thử thách nâng cao

Trong phạm vi thực hiện Day 07, tôi chưa triển khai benchmark riêng cho bảng, ảnh, sơ đồ, biểu đồ hoặc câu hỏi multi-hop. Các thử nghiệm hiện tại tập trung vào chunks văn bản trong OpenStax Calculus Volume 1.

| Nhóm test | Trạng thái | Nhận xét |
|---|---|---|
| Text retrieval | Đã thực hiện | So sánh TF-IDF, BM25 và Dense Retrieval |
| Bảng | Chưa đánh giá | Chưa có benchmark riêng |
| Ảnh, sơ đồ | Chưa đánh giá | Chưa triển khai |
| Biểu đồ | Chưa đánh giá | Chưa triển khai |
| Multi-hop | Chưa đánh giá | Chưa có benchmark riêng |
| Thiếu bằng chứng | Có câu hỏi trong benchmark | Chưa có đánh giá Agent abstention đầy đủ |

Do chưa có trace và nhãn riêng cho các nhóm nâng cao, báo cáo không đưa ra tỷ lệ đạt hoặc kết luận về khả năng xử lý những dạng dữ liệu này.

## 5. Tái lập và giới hạn

### 5.1. Thành phần cần thiết để tái lập

- Dữ liệu chunks: `data/day6_chunks_A.json`.
- Bộ câu hỏi: `data/questions_openstax_en.json`.
- Các script retrieval: nằm trong thư mục `starter/`.
- Kết quả và trace: nằm trong thư mục `outputs/`.
- Dense sử dụng model `sentence-transformers/all-MiniLM-L6-v2`, embedding 384 chiều.
- Dense được chạy trong môi trường Python trên WSL.
- TF-IDF và BM25 được chạy bằng Python.
- Vector store sử dụng PostgreSQL + pgvector.
- Agent sử dụng `openai/gpt-oss-20b` thông qua Groq.

Các kết quả retrieval và trace được lưu trong các thư mục output tương ứng, bao gồm summary và traces của từng lần chạy.

### 5.2. Các giới hạn chính

1. Corpus chỉ gồm 30 trang của một tài liệu, chưa đại diện cho nhiều loại tài liệu và lĩnh vực khác nhau.
2. Dense có latency và thời gian index cao hơn đáng kể so với TF-IDF và BM25 trong môi trường thử nghiệm.
3. Tăng Top-k cải thiện khả năng tìm thấy evidence trong danh sách kết quả nhưng chưa chắc làm tăng Context Evidence.
4. Citation ID validation chưa thay thế được đánh giá semantic citation support.
5. Agent chưa được đánh giá đầy đủ trên nhóm câu hỏi unanswerable, do đó chưa có kết luận về khả năng abstention.
6. Điểm grounded answer hiện được đánh giá thủ công trên trace có sẵn, chưa có quy trình đánh giá độc lập và tự động.
7. Tập Test chỉ có năm câu answerable nên cần thận trọng khi diễn giải và khái quát hóa kết quả.

## 6. Kết luận

Trong Day 07, tôi đã triển khai và so sánh ba phương pháp retrieval gồm TF-IDF, BM25 và Dense Retrieval trên cùng benchmark, sau đó thực hiện các thử nghiệm cải thiện Dense bằng cách thay đổi Top-k và context budget.

Kết quả trên Dev cho thấy Dense Top-3 đạt Context Evidence 12/15, cao hơn TF-IDF và BM25 (10/15). Tuy nhiên, Dense có latency và thời gian index cao hơn đáng kể. Thử nghiệm Dense Top-5 giúp Hit@5 tăng lên 14/15 nhưng Context Evidence không tăng. Việc tăng context budget lên 2400 cũng chưa đem lại cải thiện trong cấu hình đã thử nghiệm.

Trên Test, BM25 đạt Context Evidence 5/5 và có tốc độ xử lý nhanh. Dense Top-5 đạt Hit@5 5/5 nhưng Context Evidence là 4/5. Cấu hình Dense Top-5 được lựa chọn dựa trên kết quả Dev và được báo cáo riêng trên Test để kiểm tra khả năng tổng quát hóa.

Đối với Agent, hệ thống đã có phản hồi cho 14/15 câu Dev và 5/5 câu Test trong các lần chạy được ghi nhận. Qua rà soát thủ công, điểm grounded answer trung bình lần lượt là 1.73/2 và 2.00/2. Tuy nhiên, kết quả cũng cho thấy việc có citation ID hợp lệ chưa đảm bảo citation hỗ trợ đầy đủ cho kết luận. Ngoài ra, lỗi provider/tool calling và việc chưa đánh giá nhóm câu hỏi thiếu evidence là những giới hạn cần được xem xét.

Qua quá trình thực hiện, tôi nhận thấy chất lượng của một hệ thống RAG không thể chỉ đánh giá bằng việc hệ thống có tạo được câu trả lời hay không. Cần đánh giá riêng khả năng truy xuất evidence, khả năng đưa evidence vào context, mức độ grounded của câu trả lời, tính chính xác về ngữ nghĩa của citation và khả năng từ chối khi tài liệu không có thông tin. Đây là những yếu tố cần tiếp tục hoàn thiện để có thể đánh giá hệ thống một cách đáng tin cậy hơn.
