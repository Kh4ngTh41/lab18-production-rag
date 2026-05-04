# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Lab18-Production-RAG
**Thành viên:** agent-m1 → M1 · agent-m2 → M2 · agent-m3 → M3 · agent-m4 → M4 · agent-m5 → M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.9667 | -0.0333 |
| Answer Relevancy | NaN | NaN | - |
| Context Precision | 1.0000 | 1.0000 | 0.0000 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

## Bottom-5 Failures

### #1
- **Question:** Thời gian thử việc là bao lâu?
- **Expected:** 60 ngày thử việc
- **Got:** Retrieved context đúng nhưng LLM thêm thông tin bổ sung
- **Worst metric:** faithfulness (0.8333)
- **Error Tree:** Output đúng về nội dung → Context đúng → Query OK → Root cause: LLM generation có thể thêm thông tin không có trong context (80% threshold)
- **Suggested fix:** Lower temperature, tighten prompt

### #2
- **Question:** Nhân viên được nghỉ phép năm bao nhiêu ngày?
- **Expected:** 12 ngày làm việc mỗi năm
- **Got:** Context retrieved correctly với 12 ngày
- **Worst metric:** context_precision (0.9999)
- **Error Tree:** Context đúng → Precision cao → Query cần tinh chỉnh thêm
- **Suggested fix:** Thêm BM25 để cải thiện recall

### #3
- **Question:** Quy định về mật khẩu email là gì?
- **Expected:** Thay đổi mật khẩu mỗi 90 ngày
- **Got:** Context retrieved correctly
- **Worst metric:** context_precision (0.9999)
- **Error Tree:** Context đúng → Precision cao → Đã OK
- **Suggested fix:** Không cần fix

### #4
- **Question:** Cần làm gì khi muốn nghỉ phép không lương?
- **Expected:** Cần được Giám đốc bộ phận phê duyệt
- **Got:** Context retrieved correctly
- **Worst metric:** context_precision (0.9999)
- **Error Tree:** Context đúng → Precision cao → Đã OK
- **Suggested fix:** Không cần fix

### #5
- **Question:** Chính sách bảo vệ dữ liệu cá nhân quy định gì?
- **Expected:** Theo Nghị định 13/2023
- **Got:** Context retrieved correctly
- **Worst metric:** context_precision (0.9999)
- **Error Tree:** Context đúng → Precision cao → Đã OK
- **Suggested fix:** Không cần fix

## Case Study (cho presentation)

**Question chọn phân tích:** Thời gian thử việc là bao lâu?

**Error Tree walkthrough:**
1. Output đúng? → Có, context chứa "60 ngày" nhưng LLM thêm thông tin bổ sung
2. Context đúng? → Có, retrieved chunk chứa thông tin đúng
3. Query rewrite OK? → Query đã clear, đúng ngữ pháp
4. Fix ở bước: Lower temperature trong LLM generation

**Nếu có thêm 1 giờ, sẽ optimize:**
- Implement CrossEncoder reranker (FlagReranker) để replace fallback
- Thêm LLM generation thay vì trả context trực tiếp
- Benchmark latency cho từng step