Bài gốc : FedLaw : Value Aware Federated Learning With Individual Fairness and Coalition Stability

FedLaw : cơ chế khuyến khích và thuật toán tổng hợp mô hình học liên kết có nhận thức về giá trị dữ liệu tại client, sử dụng lý thuyết trò chơi hợp tác và giá trị Shapely để đánh giá đóng góp của từng client và phân bổ trọng số tổng hợp nhằm đảm bảo tính công bằng cá nhân của từng client, cũng như tính ổn định liên minh.

Lý thuyết trò chơi hợp tác và giá trị Shapely : 1 trò chơi với n người chơi, với mỗi tổ hợp hợp tác sẽ cho ra giá trị khác nhau, với v(S) : giá trị mà tổ hợp tạo ra,  ví dụ
Có 3 người chơi A B C , v(A) = 1, v(B) = 2, v(C)= 5, nhưng v(A, b) = 10 , v(B,c) = 15, v(A, B, C) = 50

Shapely value : giá trị đóng góp của 1 client dựa vào các client đã tham gia trước đó : nếu A và sau B , thì A đóng góp 8 (v(A, B) - v(B)) , nhưng nếu A vào sau B và C -> A đóng góp 35 -> đóng góp của A thay đổi : phụ thuộc vào thứ tự tham gia có thể xảy ra, như vậy, shapely value của client i theo công thức tính giá trị nhóm v(S):

$$
\phi_i^t(v^t) = \sum_{S \subseteq \mathcal{C} - \{i\}} \frac{\vert{}S\vert{}!(\vert{}\mathcal{C}\vert{} - \vert{}S\vert{} - 1)!}{\vert{}\mathcal{C}\vert{}!} \Big[ v^t(S \cup \{i\}) - v^t(S) \Big]
$$

C : tập tất cả client

$\vert{\mathcal{C}} \vert$ : số lượng client tổng

S : tổ hợp không chứa i

$\vert S \vert$ : số lượng client trong S

$v(S \cup {i})- v(S)$ : đóng góp của client i vào tổ hợp S có trước đó

$\frac{\vert{}S\vert{}!(\vert{}\mathcal{C}\vert{} - \vert{}S\vert{} - 1)!}{\vert{}\mathcal{C}\vert{}!}$ : xác suất tổ hợp S xuất hiện trước client i trong 1 thứ tự ngẫu nhiên nào đó

Shapely đảm bảo : Miễn là các anh cung cấp cho tôi một hàm giá trị nhóm $v(C)$ hợp lệ, tôi sẽ dùng công thức toán học của mình để tính ra mức phân phối thưởng công bằng tuyệt đối cho từng người mà không ai cãi được

Kí hiệu

$\vartheta$ : định giá dữ liệu

$\psi$ : đánh giá đóng góp

$\xi$ : phân bổ trọng số

Giả sử hệ thống có tập n client, ở mỗi đầu vòng giao tiếp t, client i nhận mô hình $w^{t-1}$ từ server và chạy SGD đầy đủ trên toàn bộ mạng với tập dữ liệu cục bộ $D_i$ để thu được $w_i^t$ và gửi toàn bộ lên server. Vector biến động của client i được tính bằng công thức : 

$$
\delta_i^t = w_i^t - w_i^{t-1}
$$

FedLaw so sánh các biến động của các client tại vòng giao tiếp thứ t bằng 2 metrics :

$$
dist(\delta_i^t, \delta_j^t) = \Vert \delta_i^t - \delta_j^t \Vert (1 - \cos(\delta_i^t, \delta_j^t))
$$

gồm độ lớn khoảng cách euclide và chênh lệch tương đối về hướng -> $dist(\delta_i , \delta_j)$ càng nhỏ thì cập nhật của 2 client này càng tương đồng (tương đồng về góc (hàm cos) và tương đồng về độ lớn (euclide))

Đây là độ lệch giữa 2 bản cập nhật của client i và j , chuẩn hóa về khoảng (0,1] bằng công thức

$$
\frac {dist(\delta_i, \delta_j)}{dist_{max}+1}
$$

$dist_{max}$ + 1 : tránh trường hợp nếu lỡ tồn tại 1 vòng nào đó tất cả đều cập nhật giống hệt nhau, thì ko có dist nào khác 0 -> dist_max = 0 ( lỗi chia cho 0)

Sau đó quy đổi từ khoảng cách ra điểm số giữa 2 client i và j : khoảng cách càng nhỏ điểm số giữa 2 client càng cao

$$
f(dist(\delta_i^t, \delta_j^t)) = 1 - \frac{dist(\delta_i^t, \delta_j^t)}{dist_{max} + 1}
$$

$v_t(C)$ : công thức tính giá trị nhóm : nếu 1 nhóm gồm các thiết bị bất kì C hợp sức lại thì tạo ra tổng giá trị bằng bao nhiêu, với công thức: tổng điểm giữa mọi cặp client trong C, với C là 1 nhóm con bất kì các client

$$
v^t(C) = \sum_{i, j \in C} f(dist(\delta_i^t, \delta_j^t))
$$

Và  : $\mathcal{C}$ : tập tất cả client  

Như vậy tổng điểm số của client i so với tất cả client còn lại được tính bởi công thức :

$$
r_i^t = \phi_i^t(v^t) = \sum_{j \in \mathcal{C}} f(dist(\delta_i^t, \delta_j^t))
$$

Shapely value của client i theo công thức tính giá trị nhóm v(S) cho ra mức đô đóng góp của client vào mô hình chung tại vòng giao tiếp t

Sau đó, server làm mượt đóng góp của client i với đóng góp tích lũy từ vòng trước 

$$
\hat{r}_i^t = \frac{1}{2}r_i^t + \frac{1}{2}\hat{r}_i^{t-1}
$$

Và cuối cùng là lấy tỉ lệ chuẩn hóa tuyến tính

$$
q_i^t = \frac{\hat{r}_i^t}{\sum_{k=1}^n \hat{r}_k^t}
$$

Thu được 1 vector trọng số client
