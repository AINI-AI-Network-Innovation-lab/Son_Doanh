FedCon xây dựng cơ chế tổng hợp động dựa trên đánh giá đóng góp của client, tín hiệu là khối lượng dữ liệu, chất lượng phân phối dữ liệu, giá trị Shapley và độ tương đồng mô hình (giống FedLaw + FedMMD)

FedCon yêu cầu phải biết trước phân phối của dữ liệu toàn cục (tức là phân phối của CIFAR-10, 100, ...) trước khi chia : đây là giả định mạnh, ko thực sự tốt

Ngoài ra, còn cần Proxy validation data để tính Shapley bằng lấy mẫu Monte Carlo

B1 : Đánh giá độ lệch phân phối dữ liệu

Ở những vòng giao tiếp đầu tiên, server chưa thể đánh giá chính xác đóng góp qua mô hình. FedCon tính toán độ lệch chất lượng dữ liệu của từng client so với pp toàn cục thông qua khoảng cách Kullback-leibler (KL Divergence), với i là chỉ số client, k là chỉ số nhãn

$$
DQ_i = D_{KL}(D_i \vert{}\vert{} D_g) = \sum_{k=1}^K D_i(k) \log \frac{D_g(k)}{D_i(k)}
$$

Công thức này đo lường mức độ lệch giữa phân phối nhãn dữ liệu của riêng client i ($D_i$) so với phân phối toàn cục ($D_g$). Nếu $DQ_i$ lớn, nghĩa là client đang bị mất cân bằng nặng (non-iid cao) so với mặt bằng chung , bằng cách : so sánh tỉ lệ class k ($D_i(k)$) của client i với tỉ lệ chuẩn toàn cục $(D_g(k))$

B2 : Xây dựng hàm mục tiêu và tính trọng số khởi tạo ban đầu cho các client

Với hàm mất mát cục bộ của client i , với $B_i$ là tập dữ liệu cục bộ, $l$ là hàm loss

$$
f_i(\theta) = \frac{1}{\vert{}B_i\vert{}} \sum_{l=1}^{\vert{}B_i\vert{}} l(\theta, x_l, y_l)
$$

Khi đó, hàm mục tiêu toàn cục kết hợp điều tiết độ lệch dữ liệu

$$
F_w(\theta) = \min \left(\sum_{i=1}^N w_i \cdot f_i(\theta) + \lambda \sum_{i=1}^N n_i \cdot DQ_i \right)
$$

Server muốn tìm bộ trọng số $w_k$ cho các client sao cho vừa giảm thiểu hàm mất mát cục bộ, vừa phạt (thông qua tham số điều tiết $\lambda \cdot n_k \cdot DQ_k$) nếu client có lượng dữ liệu lớn nhưng độ lệch cao (với $\theta$ là mô hình tại mỗi client)

Khi đó, công thức nghiệm cho trọng số khởi tạo ban đầu cho client i

$$
w_i = \frac{n_i \cdot e^{-\lambda \cdot DQ_i}}{\sum_{j=1}^N n_j \cdot e^{-\lambda \cdot DQ_j}}
$$

Trọng số khởi tạo của client i tỷ lệ thuận với dung lượng dữ liệu $n_i$ của client đó, điều chỉnh $e^{-\lambda \cdot DQ_i}$ dựa trên độ lệch phân phối, nếu độ lệch nhỏ -> điều chỉnh cao -> trọng số vẫn được giữ cao và ngược lại

B3 : Đánh giá đóng góp thực tế bằng Shapley value qua lấy mẫu Monte Carlo ($\phi_i$)

Shapley value gốc cho client i :

$$
\phi_i = \sum_{S \subseteq \mathcal{N} \setminus \{i\}} \frac{\vert{}S\vert{}!(N-\vert{}S\vert{}-1)!}{N!} [F(S \cup \{i\}) - F(S)]
$$

Với N là số client, S là tập hợp cient ko chứa i

Tác giả tính hàm đóng góp biên như sau : Mô hình cục bộ của các client được gộp lại thành 1 mô hình toàn cục (bằng trung bình), được đánh giá độ chính xác trên 1 tập valid tại server cho 2 mô hình $S$ và $S \cup \{i\}$ -> hàm đóng góp biên = độ lệch accuracy trên tập valid này

Xấp xỉ Shapley value qua lấy mẫu Monte Carlo để giảm độ phức tạp tính toán : 

$$
\phi_i = \frac{1}{M} \sum_{m=1}^M [F(S_m \cup \{i\}) - F(S_m)]
$$

Thay vì duyệt mọi tổ hợp với $(O(2^n))$, lấy mẫu ngâu nhiên M lần các tập con $S_m$,

$[F(S_m \cup \{i\}) - F(S\_m)]$ đo lường mức độ đóng góp của i bằng độ chênh lệch giá trị nhóm trước và sau khi i vào

B4 : Tối ưu hóa trọng số qua độ tương đồng Cosine

Độ tương đồng cosine giữa 2 mô hình client i và j ($\theta _i$ và $\theta _j$)

$$
\cos(\theta_i, \theta_g) = \frac{\theta_i \cdot \theta_g}{\Vert{}\theta_i\Vert{} \Vert{}\theta_g\Vert{}}
$$

Công thức kết hợp trọng số điều chỉnh của client i tại vòng t

$$
w_i = \alpha \cdot \phi_i + (1 - \alpha) \cdot \cos(\theta_i, \theta_g)
$$

$\phi_i$ là giá trị Shapley đóng góp của client i, còn $cos(\theta_i , \theta_g)$ đo lường độ tương đồng của client i với toàn cục, nếu độ tương đồng cao -> $1-\alpha$ làm cho $w_i$ tăng cao cùng với $\phi_i$

Bước 5: Làm mượt trọng số theo lịch sử với công thức suy giảm tuyến tính

$$
w_i^{(t)} = \left(1 - \frac{t}{T}\right) w_i^{(init)} + \frac{t}{T} w_i^{(t-spec)}
$$

với t là vòng hiện tại, T là tổng số vòng, $w_i^{init}$ là trọng số khởi tạo ban đầu cho client i ở b1 dựa trên lượng và độ lệch dữ liệu $DQ_i$, $w_i^{t-spec}$ là trọng số đóng góp tính toán được ở vòng hiện tại, $w_i^t$ trọng số sau làm mượt

Ở giai đoạn đầu, khi t nhỏ, trọng số khởi tạo chiếm thế mạnh giúp giữ ổn định mô hình để tránh sốc dữ liệu, càng về cuối, hệ thống dịch chuyển hoàn toàn sang đánh giá năng lực đóng góp thực tế của client tính ở B3
