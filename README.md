# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán HOSE (2020-2023)

Ứng dụng Streamlit này được phát triển dựa trên thuật toán tối ưu hóa danh mục đầu tư kết hợp các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands) và tối ưu hóa tham số bằng **Thuật toán bầy đàn (PSO - Particle Swarm Optimization)** thông qua thư viện `nevergrad`.

Ứng dụng giúp người dùng backtest chiến lược đầu tư với dữ liệu lịch sử của 100 mã cổ phiếu trên sàn HOSE từ năm 2020 đến 2023, tự động chọn lọc danh mục cổ phiếu tối ưu (Top-N) và phân bổ tỷ trọng theo định kỳ.

## 🚀 Tính Năng Chính
1. **Backtest Đa Chiến Lược**:
   - **RSI thuần**: Giao dịch dựa trên các vùng quá mua/quá bán của chỉ báo RSI.
   - **RSI + MACD**: Kết hợp xu hướng MACD và chỉ báo RSI để lọc tín hiệu mua/bán kèm bộ lọc xu hướng dài hạn SMA(100).
   - **RSI + Bollinger Bands**: Kết hợp dải Bollinger Bands và RSI để bắt các điểm đảo chiều.
   - **Benchmark 1/N**: Chiến lược mua và nắm giữ phân bổ đều (Equal Weight) trên toàn bộ danh mục cổ phiếu làm cơ sở so sánh.
2. **Tối Ưu Hóa Tham Số Bằng PSO**:
   - Sử dụng thuật toán PSO để tìm ra bộ tham số chỉ báo kỹ thuật tối ưu nhất (ví dụ: chu kỳ RSI, ngưỡng quá mua/bán, chu kỳ MACD...) cho từng kỳ tái cơ cấu.
3. **Phân Bổ Tỷ Trọng Linh Hoạt**:
   - Phân bổ đều (**Equal Weight**).
   - Phân bổ theo nghịch đảo biến động (**Inverse Volatility**) giúp giảm thiểu rủi ro của danh mục.
4. **Trực Quan Hóa Tương Tác**:
   - Biểu đồ tăng trưởng tài sản (Equity Curve) tương tác bằng Plotly.
   - Bảng so sánh chi tiết các chỉ số hiệu suất: Lợi nhuận gộp, CAGR, Volatility (Rủi ro biến động), Sharpe, Sortino, Max Drawdown, Calmar, Win rate.
   - Nhật ký giao dịch chi tiết và nhật ký tái cơ cấu qua từng thời kỳ.
   - Khám phá chi tiết biến động giá và các chỉ báo kỹ thuật của từng mã chứng khoán.

## 📁 Cấu Trúc Thư Mục
* `app.py`: Mã nguồn chính của ứng dụng Streamlit.
* `requirements.txt`: Danh sách các thư viện Python cần thiết.
* `HOSE_2020_2023.csv`: File dữ liệu lịch sử giá cổ phiếu sàn HOSE.
* `README.md`: Hướng dẫn sử dụng này.

## 🛠️ Hướng Dẫn Cài Đặt và Chạy Cục Bộ (Local)

### Yêu cầu hệ thống:
* Python 3.8 - 3.11 (Nevergrad hoạt động tốt nhất trên các phiên bản này).

### Các bước cài đặt:
1. Clone hoặc tải mã nguồn này về máy tính của bạn.
2. Mở terminal tại thư mục dự án và tạo môi trường ảo (khuyến nghị):
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```
3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
4. Khởi chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Trình duyệt web sẽ tự động mở ứng dụng tại địa chỉ `http://localhost:8501`.

## 🌐 Triển Khai Lên Streamlit Cloud (Deploy)
Để deploy ứng dụng của bạn lên internet miễn phí bằng Streamlit Cloud:
1. Đẩy 3 file (`app.py`, `requirements.txt`, `README.md`) cùng với file dữ liệu `HOSE_2020_2023.csv` lên một kho lưu trữ GitHub công khai (Public Repository).
2. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
3. Nhấp vào nút **New app**, sau đó chọn Repository, Branch, và file chạy chính là `app.py`.
4. Nhấn **Deploy!** và đợi hệ thống cài đặt môi trường. Ứng dụng của bạn sẽ sẵn sàng trực tuyến chỉ sau vài phút.
