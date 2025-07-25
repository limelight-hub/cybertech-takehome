# Violence Detection API Docs

## Địa Chỉ API

- **URL cơ bản**: `http://localhost:8000`
- **Tài liệu tương tác**: `http://localhost:8000/docs`
- **Tài liệu ReDoc**: `http://localhost:8000/redoc`

## Các Chức Năng Chính

### 1. Kiểm Tra Trạng Thái Hệ Thống

**Endpoint**: `GET /`

**Mục đích**: Kiểm tra xem hệ thống có hoạt động bình thường không

**Cách sử dụng**:

```
Truy cập: http://localhost:8000/
```

**Kết quả trả về**:

```json
{
  "status": "healthy",
  "message": "Violence Detection API - FastAPI Version",
  "model_loaded": true,
  "model_info": {
    "model_path": "MoBiLSTM_model.h5",
    "format": ".h5 (legacy)",
    "input_shape": [16, 64, 64, 3],
    "classes": ["NonViolence", "Violence"]
  },
  "supported_formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"],
  "timestamp": "2025-01-25T10:30:00"
}
```

**Giải thích kết quả**:

- `status`: Tình trạng hệ thống ("healthy" = khỏe mạnh)
- `model_loaded`: Mô hình AI đã được tải chưa (true/false)
- `supported_formats`: Các định dạng video được hỗ trợ

---

### 2. Phân Tích Video Phát Hiện Bạo Lực

**Endpoint**: `POST /api/detect`

**Mục đích**: Upload video để AI phân tích và phát hiện nội dung bạo lực

**Dữ liệu cần gửi**:

- `video`: Tệp video cần phân tích (bắt buộc)
- `analysis_type`: Loại phân tích (tùy chọn)
  - `"summary"`: Phân tích tổng quan (mặc định)
  - `"frame_by_frame"`: Phân tích từng khung hình

**Cách sử dụng**:

**Phương pháp 1 - Phân tích tổng quan**:

```bash
# Gửi video để phân tích nhanh
curl -X POST \
  -F "video=@video_can_phan_tich.mp4" \
  http://localhost:8000/api/detect
```

**Phương pháp 2 - Phân tích chi tiết**:

```bash
# Phân tích từng khung hình (tạo video kết quả)
curl -X POST \
  -F "video=@video_can_phan_tich.mp4" \
  -F "analysis_type=frame_by_frame" \
  http://localhost:8000/api/detect
```

**Kết quả trả về**:

_Phân tích tổng quan_:

```json
{
  "success": true,
  "file_id": "abc123-def456",
  "analysis_type": "summary",
  "result": {
    "prediction": "Violence",
    "confidence": 0.87,
    "risk_level": "High",
    "message": "Video có khả năng cao chứa nội dung bạo lực"
  },
  "processing_time_seconds": 12.5,
  "timestamp": "2025-01-25T10:35:00"
}
```

_Phân tích chi tiết_:

```json
{
  "success": true,
  "file_id": "abc123-def456",
  "analysis_type": "frame_by_frame",
  "result": {
    "prediction": "Violence",
    "confidence": 0.87,
    "risk_level": "High",
    "total_frames": 450,
    "violence_frames": 127,
    "violence_percentage": 28.2,
    "download_id": "abc123-def456"
  },
  "processing_time_seconds": 45.8,
  "timestamp": "2025-01-25T10:35:00"
}
```

**Giải thích kết quả**:

- `prediction`: Kết quả dự đoán ("Violence" = bạo lực, "NonViolence" = không bạo lực)
- `confidence`: Độ tin cậy (0-1, càng cao càng chắc chắn)
- `risk_level`: Mức độ rủi ro (Low/Medium/High)
- `violence_percentage`: Tỷ lệ phần trăm khung hình có bạo lực

---

### 3. Tải Video Đã Phân Tích

**Endpoint**: `GET /api/download/{file_id}`

**Mục đích**: Tải video đã được đánh dấu các khung hình bạo lực

**Cách sử dụng**:

```
Truy cập: http://localhost:8000/api/download/abc123-def456
```

_(Thay `abc123-def456` bằng `file_id` thực tế từ kết quả phân tích)_

**Kết quả**: Tệp video MP4 với các khung hình bạo lực được đánh dấu màu đỏ trên video

---

### 4. Kiểm Tra Trạng Thái Mô Hình AI

**Endpoint**: `GET /api/model/status`

**Mục đích**: Xem thông tin chi tiết về mô hình AI đang sử dụng

**Cách sử dụng**:

```
Truy cập: http://localhost:8000/api/model/status
```

**Kết quả trả về**:

```json
{
  "model_loaded": true,
  "model_info": {
    "model_path": "MoBiLSTM_model.h5",
    "format": ".h5 (legacy)",
    "input_shape": [16, 64, 64, 3],
    "classes": ["NonViolence", "Violence"]
  },
  "available_models": [
    {
      "path": "MoBiLSTM_model.h5",
      "size_mb": 45.2,
      "format": "h5 (legacy)"
    }
  ],
  "supported_formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv"]
}
```

---

### 5. Tải Mô Hình AI Mới

**Endpoint**: `POST /api/model/load`

**Mục đích**: Thay đổi mô hình AI đang sử dụng

**Dữ liệu cần gửi**:

- `model_path`: Đường dẫn đến file mô hình (mặc định: "MoBiLSTM_model.h5")

**Cách sử dụng**:

```bash
curl -X POST \
  -F "model_path=MoBiLSTM_model.keras" \
  http://localhost:8000/api/model/load
```

---

## Mã Lỗi Thường Gặp

### Thành Công

- **200**: Yêu cầu thành công
- **Kết quả**: Dữ liệu JSON như mô tả ở trên

### Lỗi Người Dùng

- **400**: Định dạng file không hỗ trợ
  - _Giải pháp_: Chỉ sử dụng các file: mp4, avi, mov, mkv, wmv, flv
- **422**: Dữ liệu gửi lên không đúng định dạng
  - _Giải pháp_: Kiểm tra lại cách gửi file và tham số

### Lỗi Hệ Thống

- **500**: Lỗi xử lý bên trong
  - _Nguyên nhân_: Mô hình AI chưa được tải, file video bị hỏng, hoặc lỗi server
  - _Giải pháp_: Kiểm tra trạng thái hệ thống tại endpoint `/`

---

## Hướng Dẫn Sử Dụng Đơn Giản

### Bước 1: Kiểm tra hệ thống

```
Truy cập: http://localhost:8000/
Xem kết quả có "status": "healthy" không
```

### Bước 2: Upload video

```
Sử dụng tool như Postman hoặc curl
Chọn file video từ máy tính
Gửi đến: http://localhost:8000/api/detect
```

### Bước 3: Xem kết quả

```
Kiểm tra trường "prediction" và "confidence"
Nếu muốn tải video đã phân tích, dùng "file_id"
```

---

## Ví Dụ Thực Tế

### Trường hợp 1: Video an toàn

```json
{
  "prediction": "NonViolence",
  "confidence": 0.92,
  "risk_level": "Low",
  "message": "Video không chứa nội dung bạo lực"
}
```

### Trường hợp 2: Video có bạo lực

```json
{
  "prediction": "Violence",
  "confidence": 0.87,
  "risk_level": "High",
  "message": "Video có khả năng cao chứa nội dung bạo lực"
}
```

### Trường hợp 3: Không chắc chắn

```json
{
  "prediction": "NonViolence",
  "confidence": 0.55,
  "risk_level": "Medium",
  "message": "Cần xem xét thủ công để đảm bảo"
}
```

---

## Lưu Ý Kỹ Thuật

1. **Kích thước file**: Không giới hạn cụ thể, nhưng file càng lớn xử lý càng lâu
2. **Thời gian xử lý**:
   - Phân tích tổng quan: 10-30 giây
   - Phân tích chi tiết: 30-120 giây (tùy độ dài video)
3. **Định dạng hỗ trợ**: MP4, AVI, MOV, MKV, WMV, FLV
4. **Độ chính xác**: Khoảng 85-95% tùy thuộc vào chất lượng video

---

## Hỗ Trợ

Nếu gặp vấn đề, hãy kiểm tra:

1. Hệ thống có đang chạy không: `http://localhost:8000/`
2. Mô hình AI đã được tải chưa: `http://localhost:8000/api/model/status`
3. Định dạng video có được hỗ trợ không
4. Kết nối mạng có ổn định không

---
