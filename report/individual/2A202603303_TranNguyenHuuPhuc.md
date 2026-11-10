# Báo cáo cá nhân — Ngày 2 Object Detection

Đổi tên file theo `MSSV_HoTen.md`, ví dụ `20241234_NguyenVanAn.md`. Điền kết quả thật, xóa dòng hướng dẫn khi nộp. Xem [quy định nộp](../../SUBMISSION.md).

## 1. Thông tin học viên

- Họ tên: Trần Nguyễn Hữu Phúc
- MSSV:2A202603303
- Email: hhuuphucc@gmail.com
- Lớp: 2B-LAB-E403
- Link repo bài làm: https://github.com/hhuuphucc/K5-L2-DAY2-TranNguyenHuuPhuc-2A202603303-ObjectDetection.git
- Link Colab đã chạy: https://colab.research.google.com/github/hhuuphucc/K5-L2-DAY2-TranNguyenHuuPhuc-2A202603303-ObjectDetection/blob/main/notebooks/day2_annotation_to_yolo.ipynb

## 2. Schema và mini guideline

Dẫn tới [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md). Tóm tắt ba ca mơ hồ đã quyết định, cách áp dụng schema và phần cần giảng viên làm rõ.
- Ca 1 (Xe 16 chỗ/Minivan):** Rất dễ nhầm lẫn giữa `van` và `bus`. Cần thống nhất phân loại theo kích thước/mục đích sử dụng. 
- Ca 2 (Xe ở góc xa/bị che khuất trên 70%):** Quyết định loại bỏ không gán nhãn nếu phần hiển thị quá nhỏ, không đủ đặc trưng nhận diện lớp.
- Ca 3 (Xe máy/xe đạp):** Theo schema không thuộc 4 lớp yêu cầu, quyết định không vẽ box để tránh nhiễu dữ liệu.
## 3. Annotation và self-QC

- Số file nhãn train: 50
- Kết quả checker và lỗi đã xử lý: Script `tools/check_yolo_labels.py` báo Passed 50/50 ảnh. Đã xử lý các lỗi ban đầu về cấu trúc file (format YOLO) và đồng bộ tên ảnh/file text.
- Danh sách ít nhất 10 ảnh đã rà bằng mắt: `drive_001.jpg` đến `drive_010.jpg`.
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: Phát hiện lỗi vẽ dư box (FP) ở các vật thể méo mó/ở quá xa và thiếu box (FN) ở các khu vực rìa ảnh. Đã xóa các box không hợp lệ và vẽ bổ sung.
- Commit khóa nhãn ban đầu: `9a9f119`

## 4. Gold QC và rework

| Chỉ số | Trước sửa | Sau sửa nếu có |
|---|---:|---:|
| TP | | 478 |
| FP | | 137 |
| FN | | 175 |
| Precision | | 0.7772 |
| Recall | | 0.7320 |
| F1 | 0.5683 | 0.7539 |
| mean_iou_of_matches | 0.7876 | 0.7935 |
| passed | false | true |

Dẫn tới [outputs/annotation_evaluation.json](../../outputs/annotation_evaluation.json) trong repo. 
**Ghi chú rework:** Ở lần chạy đầu tiên, điểm F1 chỉ đạt 0.5683 do tỷ lệ FP (274) và FN (285) cao. Sau khi nhận được sự cho phép từ lab coach hạ tiêu chuẩn `required_f1` xuống 0.5, tôi đã điều chỉnh script kiểm tra. File đánh giá báo `passed: true` nên không tiến hành gán lại toàn bộ 50 ảnh.
## 5. Kết quả YOLO

- Runtime, GPU/CPU, phiên bản Ultralytics và Torch: Colab Runtime: **CPU** (Intel Xeon CPU @ 2.20GHz) | Ultralytics: **8.4.208** | Torch: **2.14.1+cpu**.
- Các tham số thay đổi so với notebook nếu có và lý do: Không thay đổi tham số huấn luyện của notebook, chỉ chỉnh sửa biến `required_f1 = 0.5` trong `evaluate_annotations.py` theo hướng dẫn của coach để bypass lỗi QC.
- Đường dẫn best checkpoint trong phiên chạy: `/content/Day2-Lab/outputs/yolo26_runs/vehicle_annotation/weights/best.pt`
  
| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.323 |
| Recall | 0.296 |
| mAP50 | 0.320 |
| mAP50–95 | 0.198 |

- Ảnh dự đoán: [assets/eval_predictions/visuals/drive_053.jpg](../../report/individual/assets/eval_predictions/visuals/drive_053.jpg) (Ví dụ hình ảnh minh họa xe `bus` và `truck` bị nhận diện sai/sót).
- Log huấn luyện: [assets/yolo26_runs/vehicle_annotation/results.csv](../../report/individual/assets/yolo26_runs/vehicle_annotation/results.csv)
  
Trả lời bốn câu hỏi cuối notebook:

1. Lớp nào recall thấp nhất? Ảnh nào minh họa?
- Lớp `bus` có Recall thấp nhất (Recall = 0.000). Minh họa qua ảnh `assets/eval_predictions/visuals/drive_053.jpg` (ảnh chứa 2 bus nhưng model không nhận diện được).
2. Nếu mAP50 cao nhưng mAP50–95 thấp, cần kiểm điều gì trên box?
- Cần kiểm tra độ ôm sát (tightness) của bounding box. mAP50-95 yêu cầu độ chính xác về tọa độ (IoU từ 0.5 đến 0.95) khắt khe hơn. Nếu mAP50-95 thấp, chứng tỏ các box vẽ lỏng lẻo, lẹm vào nền hoặc không bao phủ toàn bộ vật thể.
3. Lỗi class nào có thể liên quan guideline? Dữ kiện nào hỗ trợ nhận xét?
- Lỗi phân loại giữa `van`, `bus`, và `car`. Dữ kiện: Chỉ số của `bus` và `van` cực kỳ thấp (Bus mAP50 = 0.0155, Van mAP50 = 0.121). Guideline chưa đủ chi tiết để rạch ròi biên giới giữa minivan (thuộc `van`), minibus (thuộc `bus`), hoặc SUV (thuộc `car`), khiến sinh viên gán nhãn thiếu nhất quán, dẫn đến model bị nhiễu. Sự mất cân bằng dữ liệu (chỉ có 2 bus, 12 van) cũng làm trầm trọng thêm vấn đề này.
4. Vì sao không dùng 10 ảnh eval để train? Vai trò validation của chúng trong notebook là gì?
- Không dùng ảnh eval để huấn luyện nhằm giữ một tập dữ liệu "mù" hoàn toàn độc lập, tránh hiện tượng overfitting (học thuộc lòng). Vai trò của chúng trong notebook là làm tập **Validation**: đo lường hiệu suất khách quan của mô hình sau mỗi epoch và giúp thuật toán quyết định chọn lưu lại epoch có hiệu suất tốt nhất (`best.pt`).

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Khởi tạo script chấm điểm | Báo lỗi cú pháp Python trong file `evaluate_annotations.py` | Sửa lại các dấu `-` và `*` bị thiếu do copy/paste | Script chạy thành công |
| QC với Gold Label | `passed: false` do F1 (0.56) thấp hơn yêu cầu (0.8) | Hỏi AI xác định lỗi dư/sót; sửa code hạ ngưỡng `min_f1` xuống 0.5 theo yêu cầu của Coach | Báo cáo Evaluation trả về `passed: true` |
| Train YOLO trên Colab | Runtime mặc định là CPU khiến inference chậm | Kích hoạt lại session nhưng hết lượt T4 GPU | Chấp nhận train bằng CPU (Intel Xeon), tốn ~0.654 hours |
## 7. Tự đánh giá theo rubric

Đối chiếu [RUBRIC.md](../../RUBRIC.md): phần nào đã có kết quả, phần nào còn thiếu hoặc chưa kiểm được. Không tự nhận hoàn thành bước chưa chạy.

- Hoàn thành và định dạng YOLO: Đạt. 50 file nhãn chuẩn YOLO.
- Đầy đủ vật thể & Bám sát box: Cần cải thiện. Vẫn còn lỗi FN và FP dẫn đến F1 ban đầu là 0.7539.
- Gold-set annotation QC: Đạt, nhờ sự hỗ trợ hạ ngưỡng từ coach.
- Pipeline YOLO và phân tích: Đạt. Hoàn thiện toàn bộ quá trình training, trích xuất folder `assets`, report chỉ số metrics và trả lời đầy đủ 4 câu hỏi.
## 8. Công cụ AI và tài liệu tham khảo

Ghi công cụ đã dùng, dùng ở bước nào và phần đã tự kiểm tra; không dùng thì ghi rõ. Liệt kê nguồn ngoài repo được sử dụng. Không ghi mật khẩu/token hoặc đính kèm gold train.

- Công cụ AI: Sử dụng Gemini để phân tích logic tính IoU, sửa lỗi cú pháp Python trong script `.py`.
- Tài liệu: Đọc hướng dẫn từ `CHECKPOINTS.md`, `GUIDE.md` và `SUBMISSION.md` của Ban tổ chức.
