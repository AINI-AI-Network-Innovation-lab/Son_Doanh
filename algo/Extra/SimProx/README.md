$w_i$ : mô hình tại client i ở vòng hiện tại

$w_i^{prev}$ : mô hình tại client i ở vòng trước đó

$w^{global}$ : mô hình toàn cục tại server

$\sigma$ : khoảng cách trung bình giữa tất cả các cặp mô hình tại các client

S : ma trận tương đồng giữa các client

$\lambda$ : hyperparams : điều chỉnh tỷ trọng giữa 2 độ đo tương đồng

$g_i$ : độ dời tiệm cận của mô hình tại client i (proximal gradient norm)

$\theta_i$ : trọng số phân bổ cho client i

m : số client

$D_i$ : dữ liệu riêng của client i để có mô hình mới $w_i$

B1 : Định lượng độ phân tán tổng thể của các mô hình cục bộ tại vòng giao tiếp thứ t = trung bình khoảng cách Euclide giữa 2 model client bất kì

$$
\sigma = \frac{1}{C^2_m} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \Vert w_i - w_j \Vert_2
$$

Trong đó $C^2_m$ : số cặp client không lặp lại có thể tạo ra từ m client. $\sigma$ càng nhỏ -> mô hình cục bộ càng gần nhau và ngược lại đang phân kì mạnh do non-iid ($O(n^2)$)

2 metrics thành phần để xây dựng ma trận tương đồng chéo S , với $S_{ij}$ là độ tương đồng Gaussian giữa 2 clieint i và j

$$
S_{ij}^{Gaussian} = \exp\left(-\frac{\Vert w_i - w_j \Vert_2^2}{2\sigma^2}\right)
$$

(Đưa về phân phối chuẩn Gaussian với phương sai là $\sigma$), khoảng cách càng nhỏ thì độ tương đồng điểm càng cao và ngược lại

Và độ tương đồng Cosine đo độ thống nhất hướng giữa mô hình của 2 client

$$
S_{ij}^{Cosine} = \frac{\langle w_i, w_j \rangle}{\Vert w_i \Vert_2 \Vert w_j \Vert_2}
$$

2 metrics này hỗ trợ nhau : tương đồng Gaussian có thể đánh giá rất cao 2 mô hình gần nhau nhưng ngược hướng , và ngược lại Cosine có thể đánh giá cao 2 mô hình tương đồng hướng nhưng rất xa nhau, khi đó, ma trận điểm tương đồng S

$$
S_{ij} = \lambda S_{ij}^{Cosine} + (1 - \lambda) S_{ij}^{Gaussian}
$$

Để thích ứng động với mức độ lệch chuẩn dữ liệu qua từng vòng, lambda được cập nhật tự động dựa trên độ đồng thuận chung. Gọi $s_i$ là độ tương đồng Cosine giữa mô hình cục bộ $w_i$ và toàn cục $w^{global}$ -> độ tương đồng trung  bình toàn hệ thống là : $\bar {s} = \frac{1}{m} \sum_{i=1}^{m}s_i$ . Với ngưỡng phân tán $\tau$ (hyperparams) và giá trị gốc $\lambda_0$ (từ 0.6 -> 0.7) , $\lambda$ được điều chỉnh với công thức :

$$
\lambda = \begin{cases} \lambda_0 \cdot \left(\frac{\bar{s}}{\tau}\right) & \text{nếu } \bar{s} < \tau \\ \lambda_0 & \text{ngược lại} \end{cases}
$$

Khi $\bar{s} < \tau$ : xu hướng chung các mô hình cục bộ bị lệch hướng so với mô hình toàn cục , hệ thống hạ $\lambda$ tại vòng t xuống để tăng tỉ trọng của $S_{ij}^{Gaussian}$, ít phụ thuộc hơn vào tương đồng Cosine để tránh trôi dạt mô hình global

Ngoài so sánh độ tương đồng chéo, SimProx đo lường mức độ biến động của client i với chính nó ở vòng trước, cho biết độ dời mô hình cục bộ trong vòng giao tiếp hiện tại. Giả định $g_i$ nhỏ đồng nghĩa đã nằm gần điểm tối ưu toàn cục và cập nhật ổn định và ngược lại : do dữ liệu cực đoan kéo mô hình đi xa -> nếu dùng mô hình này nhiều sẽ gây drift

$$
g_i = \Vert w_i - w_i^{prev} \Vert_2
$$

Trọng số thô của client i được tính bằng cách kết hợp độ dời $g_i$ và độ tương đồng trung bình của client i với m-1 client còn lại

$$
\theta_i = \exp(-g_i) \left( 1 + \frac{\sum_{j \neq i} S_{ij}}{m - 1} \right)
$$

Khi độ dời càng lớn -> $\theta$ càng nhỏ

Sau đó chuẩn hóa tuyến tính lần 1 để đưa tổng trọng số của các client về 1, thu hẹp biên độ về [0,1], ngăn trường hợp $\theta_i$ quá lớn làm bùng nổ hàm softmax

$$
\hat{\theta}_i = \frac{\theta_i}{\sum_{k=1}^m \theta_k}
$$

Áp dụng softmax để làm mượt phân bổ trọng số

$$
\theta_i^{final} = \text{softmax}(\hat{\theta})_i = \frac{e^{\hat{\theta}_i}}{\sum_{j=1}^m e^{\hat{\theta}_j}}
$$

Và tổng hợp mô hình toàn cục mới :

$$
w^{global} = \sum_{i=1}^m \theta_i^{final} w_i
$$

Lý do chọn cơ chế đánh giá mô hình và phân bổ trọng số như vậy : *Nếu hai mô hình cục bộ cùng huấn luyện trên các tập dữ liệu bị lệch (skewed/Non-IID) khác nhau nhưng sau khi học lại hội tụ về cùng một cấu trúc tham số tương đồng nhau, thì phần thông tin giao thoa đó chính là tri thức cốt lõi mang tính khái quát hóa (generalization) cao nhất cho bài toán chung*<sup></sup>. Do đó, **độ tương đồng trung bình của mô hình $w_i$ với tập thể** được dùng làm thước đo cho **giá trị đóng góp mặt tri thức của client**<sup></sup>, kết hợp cùng **hệ số suy giảm tiệm cận $\exp(-g_i)$** đóng vai trò như bộ giảm chấn giúp kìm hãm các mô hình biến động quá khích<sup></sup>. Vì mô hình toàn cục là tổ hợp lồi từ tất cả mô hình cục bộ ($w^{global} = \sum \theta_i^{final} w_i$)<sup></sup>, nên một client vừa có mô hình $w_i$ đồng thuận cao với đại đa số các client khác, vừa có độ dời $g_i$ nhỏ quanh điểm tối ưu (tức nhận trọng số $\theta_i^{final}$ lớn) thì cũng chính là client đang giữ cho mô hình toàn cục hội tụ nhanh và ổn định nhất<sup></sup>.
