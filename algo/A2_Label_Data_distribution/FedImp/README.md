FedImp : Federated Impurity Weighting

Ý tưởng tổng quát :

FedAvg chỉ chia trọng số dựa trên số lượng mẫu dữ liệu $D_i$

FedAdp gán trọng số dựa trên góc lệnh gradient của client i so với trung bình gradient toàn cục (ai cùng hướng với đám đông thì trọng số cao), nhưng FedImp chỉ ra kịch bản : Giả sử có 10 client, trong đó có 3 client có dữ liệu cân bằng chuẩn (đủ class), nhưng 7 client còn lại đều mất cân bằng giống hết nhau, khi đó 7 client bị lệch này áp đảo hoàn toàn -> đạo hàm toàn cục bị kéo về hướng của 7 client bị lệch => FedAdp tưởng đấy là hướng chuẩn nên chia đều trọng số cho cả 7 client lệch rồi lại phát trọng số thấp cho 3 client đều -> mô hình sập

Ý tưởng của FedImp : đo độ giàu thông tin (Impurity/Entropy) của từng client

Thay vì nhìn vào gradient hay khoảng cách tham số (rất dễ bị lừa khi đa số các máy cùng bị non-iid), FedImp nhìn vào độ đa dạng nhãn dữ liệu cục bộ tại từng client qua Entropy : Client nào có nhãn đa dạng , tỉ lệ đồng đều => Entropy về gần 1 và ngược lại cho client bị lệch, như vậy , client nào có entropy cao -> xứng đáng được trọng số cao hơn và ngược lại

B1 : Tính entropy tại từng client với công thức entropy với cơ số C, K là số class tại client i (khi lấy logC với C là số class, Entropy tự nén vào trong khoảng [0,1]

$$
S_i = -\sum _{k=1}^K p_{i,k} log_C(p_{i,k})
$$

B2 : Tính trọng số tổng hợp cho client i với nhiệt độ $\tau > 0$

$$
\psi_i(t) = \frac{D_i \cdot e^{\frac{S_i}{\tau}}}{\sum_{j=1}^{K} D_j \cdot e^{\frac{S_j}{\tau}}}
$$

Trong đó, $D_i$ là số lượng mẫu tại client i, $S_i$ là entropy tại client i, $e^{\frac{S_i}{\tau}}$ khuyếch đại đóng góp của client qua Entropy.

Nhiệt độ $\tau$ nhỏ (-> 0) : thành phần khuếch đại đánh giá cao hơn các client có entropy cao ($\frac{S_i}{\tau}$ tăng nhanh hơn), và ngược lại , $\tau$ rất lớn : hiệu ứng của Entropy biết mất, bài toán trở về thành FedAvg
