Bài báo gốc : FedSL: A communication-efficient federated learning with split layer aggregation

FedSL(Federated split layers): tên gọi chung của bộ khung thuật toán học liên kết tiết kiệm chi phí truyền thông bằng cách chia tách các tầng layer của mô hình, bao gồm FedASL và FedGSL

FedGSL (Group split Layers) : Giả sử mô hình CNN gồm 6 tầng tích chập (1->6), chia thành 3 nhóm : nhóm 1 gồm tầng 1 tầng 2, nhóm 2 gồm tầng 3 4, nhóm 3 gồm tầng 5 6. Có 9 client, ở mỗi vòng giao tiếp chia ra 3 nhóm, chạy train và backward trên toàn mạng bình thường, nhưng khi gửi, các client nhóm 1 chỉ gửi trọng số tầng 1 2, tương tự với nhóm 2 và 3. Máy chủ lấy trung bình cộng của các tầng 1-2 từ các thiết bị nhóm 1 để ra tầng 1 và 2 mới, tương tự cho 2 và 3, rồi ghép lại, đắp thêm bộ phân loại và gửi lại client

* FedASL (Adaptive split layers) : Tương tự GSL , ở mỗi đầu vòng, client nhật 1 mô hình hoàn chỉnh từ server và chạy forward backward đầy đủ trên tất cả tầng mạng. Nhưng ASL ko chia ngẫu nhiên các tầng thành p nhóm độc lập, mà ban đầu, ASL coi 6 tầng như 6 nhóm riêng biệt. Sau khi train xong, client k không upload ngay tầng tích chập nào, mà lấy từng tầng so với tầng tương ứng của model toàn cục nhận đầu vòng để tính số biến động bằng chuẩn euclid, với $l$ là số hiệu layer

$$
c_{l,k}^t = \Vert{}w_{l,k}^t - w_{l,k}^{t-1}\Vert{}
$$

* Như vậy, tại mỗi vòng, client k chỉ cần gửi 1 vector b chiều (cho b tầng) lên server, mỗi chiều là khoảng cách euclide của tầng đó so với trước khi train, đính kèm với các lớp phân loại của nó. Server sẽ chia làm 2 kiểu vòng, gồm vòng chẵn (pha Max) và vòng lẻ (pha Min), tại đó

Pha Max : Chọn các thiết bị có khoảng cách trong vòng cao nhất, tầng này ở thiết bị đó biến động nhiều nhất, cần chọn để hội tụ nhanh

Pha Min : chọn các thiết bị có khoảng cách trong vòng thấp nhất, tầng này ở thiết bị đó ít biến động, dùng để tinh chỉnh cho mô hình)

* Ưu tiên chọn những thiết bị lâu rồi chưa được chọn để gửi ở tầng $l$ : (FindOldestClients)

***Thuật toán FindOldestClients*** : $n$ : số thiết bị tham gia vòng giao tiếp t,

$s_l[i]$ : vòng cuối cùng mà client i được gọi để đóng góp cho tầng $l$

$n_s$ : hyperparams : số lượng thiết bị cần tải tầng $l$ lên ở mỗi vòng giao tiếp

Bước 1 : Sau khi sắp xếp khoảng cách tầng $l$ theo tăng dần (vòng lẻ) hoặc giảm dần (vòng chẵn) -> Cắt nửa danh sách : $n_h = [n/2]$, lấy các thiết bị đầu danh sách

Bước 2 : Tính số vòng trung bình để 1 thiết bị trong nhóm $n_h$ được gọi 1 lần : Trong nhóm $n_h$ client được lọc ở trên, chọn ra $n_s$ client để gửi -> về mặt xác suất , cần $[n_h/n_s]$ vòng để 1 thiết bị nào đó được chọn để gửi : chu kì vòng quay cơ bản

Bước 3 : Nếu sau hơn 5 chu kì vòng quay cơ bản liên tiếp mà client i (thuộc nhóm n_h được lọc, ko quét hết tất cả) không được gọi vào đóng góp cho tầng $l$ -> thiết bị đang bị bỏ rơi -> chọn ngay client i này vào để gửi và cập nhật $s_l[i]$ cho client i, với công thức :

$$
t-s_l[i] > 5 \times \lceil n_h/n_s \rceil
$$

với t là số hiệu vòng hiện tại

Nếu sau khi bốc các client bị bỏ rơi, mà ko đủ $n_s$ client yêu cầu, quay lại $n_h$ client được lọc trước đó, bốc từ đầu danh sách đến khi đủ, và cũng cập nhật $s_l[i]$ cho các client được chọn này

Sau đó : yêu cầu upload theo các client đã chọn : Client A chỉ gửi tầng 0, B gửi tầng 1 và 3 -> tiết kiệm chi phí truyền thông

* Server tổng hợp từng tầng : lấy trung bình có trọng số các tham số do các thiết bị được chọn gửi tầng đó gửi lên $w_l^{temp}$. Làm mượt bằng Decay-Moving Average : Lấy tầng mới vừa tổng hợp, trộn với tầng cũ của vòng trước theo hệ số suy giảm $\beta ^t$, được sinh ra theo công thức :

$$
\beta^t = \max\left(\beta_{min}, \beta_0^{\lfloor \frac{t}{s} \rfloor}\right)
$$

Với $\beta _0$ là hệ số suy giảm xuất phát ban đầu, $s$ là chu kì bước nhảy suy giảm, và $\beta _{min}$ là ngưỡng sàn tối thiểu

với công thức cập nhật trọng số tầng $l$ mới :

$$
w_l^t= \beta ^tw^{temp}_l + (1-\beta ^t)w_l^{t-1}
$$

Sau đó lấy trung bình bộ phân loại từ tất cả thiết bị tham gia vòng, ghép nối lại và gửi lại client

Mở rộng : 

Như vậy, đầu ra tại mỗi vòng giao tiếp có thể biểu diễn dưới dạng ma trận 2 chiều :


$$
M^0 = \begin{matrix} \text{Client 1:} & [0.50, & 0.00, & 0.333] \\ \text{Client 2:} & [0.50, & 0.50, & 0.333] \\ \text{Client 3:} & [0.00, & 0.50, & 0.333] \end{matrix}
$$

Với hàng i , cột j là tỉ lệ đóng góp của client i vào layer thứ j, hoặc nén thành 1 chiều bằng cách nhân với 1 ma trận tỉ lệ quan trọng của layer
