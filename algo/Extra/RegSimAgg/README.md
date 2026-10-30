RegSimAgg tổng hợp trọng số có điều chuẩn dựa trên mức độ tương đồng, gồm 2 giai đoạn : chọn client và tổng hợp trọng số tại server

Chiến lược chọn client : Thay vì cho toàn bộ client đóng góp, sử dụng cửa sổ trượt (sliding window) trên danh sách chỉ mục ngẫu nhiên để tối ưu chi phí giao tiếp

Bước 1 : Xáo trộn ngẫu nhiên thứ tự của tất cả các máy khách

Bước 2 : Ở mỗi vòng huấn luyện, chỉ chọn 1 tập con các client bằng cách trượt cửa sổ dọc theo danh sách đã trộn

Bước 3 : Khi cửa sổ chạy hết 1 lượt, xáo trộn lại danh sách và lặp lại

Cơ chế đảm bảo mọi máy khách đều được tham gia huấn luyện với số lần tương đương nhau theo chu kì đều đặn

# Thuật toán tổng hợp trọng số có điều chuẩn

Tại vòng huấn luyện thứ t , các client thuộc tập được gọi đóng góp gửi bộ tham số mô hình cục bộ $w_i$ về server, máy chủ thực hiện các bước sau để cập nhật mô hình toàn cục tại vòng t

Bước 1 : tính avg không trọng số của các client gửi lên ở vòng t

$$
\hat{p} = \frac{1}{\vert{}C^r\vert{}} \sum_{i \in C^r} p_i
$$

Bước 2 : Tính trọng số tương đồng : độ tương đồng của từng client c được chọn thông qua khoảng cách nghịch đảo từ mô hình của client đến mô hình trung bình , với $\epsilon$ nhỏ tùy ý để tránh divide by zero

$$
sim_c = \frac{\sum_{i \in C^r} \vert{}p_i - \hat{p}\vert{}}{\vert{}p_c - \hat{p}\vert{} + \epsilon}
$$

Sau đó, chuẩn hóa giá trị này để thu được trọng số tương đồng $u_c$ của client c

$$
u_c = \frac{sim_c}{\sum_{i \in C^r} sim_i}
$$

Các client có mô hình gần với trung bình sẽ có điểm cao hơn (nghịch đảo tương đồng) -> trọng số tương đồng cao hơn

Bước 3 : Để điều chỉnh sự chênh lệch về quy mô dữ liệu của các client, thuật toán tính thêm $v_c$ dựa trên số lượng mẫu của client c, ưu tiên các máy khách đóng góp nhiều hơn (train trên nhiều dữ liệu hơn)

$$
v_c = \frac{N_c}{\sum_{i \in C^r} N_i}
$$

Bước 4 : Tổng hợp trọng số kết hợp $w_c$

Kết hợp cả trọng số tương đồng với mô hình trung bình và số lượng mẫu dữ liệu để tạo thành trọng số tổng hợp của client c

$$
w_c = \frac{u_c + v_c}{\sum_{i \in C^r} (u_i + v_i)}
$$

Bước 5 : Điều chuẩn trọng số (nếu t > $\tau$ cho trước)

Khi quá trình huấn luyện vượt qua 1 số vòng giao tiếp t nhất định, trọng số $w_c$ sẽ được điều chuẩn dựa trên trung bình mức độ biến thiên giữa mô hình ở vòng trước $p_c^{prev}$ và $p_c$ của các client được chọn. Khi đó trung bình biến thiên mô hình của các client càng lớn thì trọng số càng nhỏ lại, kiềm chế lại việc trọng số bị biến thiên tại các client

$$
w_c = \frac{w_c}{\frac{1}{\vert{}C^r\vert{}} \sum_{i \in C^r} \vert \vert p_i^{prev} - p_i \vert \vert}_2
$$

Bước 6 : Tổng hợp mô hình toàn cục mới

$$
p^m = \frac{1}{\vert{}C^r\vert{}} \cdot \sum_{i \in C^r} (w_i \cdot p_i)
$$

Đặc điểm kĩ thuật

* **Xử lý hoàn toàn tại máy chủ (Server-side adaptability):** Khác với các phương pháp như FedProx (đòi hỏi điều chuẩn trực tiếp trong quá trình huấn luyện tại máy khách), RegSimAgg thực hiện toàn bộ việc tính toán và điều chuẩn ở phía máy chủ mà không can thiệp vào quy trình huấn luyện cục bộ của từng viện nghiên cứu/bệnh viện<sup></sup>.
* **Không phụ thuộc ngưỡng thủ công:** Thuật toán tự động điều chỉnh trọng số một cách thích ứng ở mỗi vòng mà không yêu cầu người dùng phải định nghĩa trước các ngưỡng cắt (thresholds) cố định như phương pháp làm mịn hàm mũ (exponential smoothing)<sup></sup>.
* **Hạn chế đối với dữ liệu động:** Thuật toán hiện giả định số lượng bệnh nhân ($N\_c$) tại mỗi máy khách là cố định xuyên suốt quá trình huấn luyện<sup></sup>. Trong môi trường thực tế, nếu một bệnh viện bổ sung thêm dữ liệu bệnh nhân mới giữa các vòng liên kết, phân phối dữ liệu cục bộ có thể thay đổi và làm đảo lộn tương quan độ tương đồng giữa các máy khách<sup></sup>.
