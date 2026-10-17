$w^t$ : mô hình toàn cục server gửi xuống ở đầu vòng t

$w_k^{t+1}$ : mô hình client k tính ra được và gửi lên server ở cuối vòng t

$w_k^{t, s}$ : mô hình tại client k ở vòng giao tiếp t và bước s → $w_k^{t, 0}$ = $w^t$ (model trước bước đầu tiên = model nhận từ server ở đầu vòng), w_k_(t, S_k) = w_k_t+1 (model sau bước train cuối = model gửi lên server ở cuối vòng)

w : biến định nghĩa hàm

FedProx dùng h_k thay cho F_k tại bài toán gốc, h_k(w, w_t) là hàm cần tối ưu của client k tại vòng lặp thứ t, với công thức :

$$
h_k(w;w^t)=F_k(w)+\frac\mu2\|w-w^t\|^2
$$

với F_k(w) là loss gốc của các điểm dữ liệu tại client k , phần còn lại là hàm phạt L2 : không cho w đi quá xa so với w_t nhận từ server ở đầu vòng lặp

Khi đó, gradient của h_k(w, w_t) :

$$
\nabla h_k(w;w^t)=\nabla F_k(w)+\mu\,(w-w^t)
$$

với

$$
\mu\,(w-w^t)
$$

là vector chỉ từ w → w_t

- Tại mỗi bước s tại client k, thực hiện cập nhật w theo công thức :

$$
w_k^{t,s+1}=w_k^{t,s}-\lambda\big[\nabla h_k(w_k^{t,s}, w^t)\big]
$$

Tức là : w_k được kéo về hướng cục bộ, đồng thời bị kéo theo hướng tới w toàn cục được cấp ở đầu vòng giao tiếp

Số bước tại thiết bị k (S_k) nhiều hay ít phụ thuộc vào sức mạnh của client k

- Server tổng hợp trung bình các model thành model cho vòng giao tiếp tiếp theo

  $$
  ^{t+1}=\frac1K\sum_{k\in S_t}w_k^{t+1}
  $$

Mỗi client k không tìm min h_k đến cùng ở mỗi vòng lặp giao tiếp, chỉ chạy S_k bước, và đo mức độ đã giải bằng

$$
\gamma_k^t\in[0,1]
$$

và :

$$
\big\|\nabla h_k\big(w_k^{t+1};w^t\big)\big\|\le\gamma_k^t\,\big\|\nabla h_k\big(w^t;w^t\big)\big\|
$$

hay : độ lớn gradient lúc dừng chạy phải ≤ gamma * độ lớn gradient lúc bắt đầu chạy → gamma càng nhỏ thì giải chính xác & ngược lại

- Ưu điểm : kéo cho w cục bộ không quá xa w toàn cục , tiết kiệm chi phí truyền thông
- Nhược điểm : Khi mà w được giữ ko cho đi quá xa cực tiểu toàn cục, tại các client bị thiên lệch dữ liệu, việc dự đoán sẽ kém hơn , đồng thời, chỉ giữ k đi quá xa toàn cục chứ k ngăn đi tới cực tiểu cục bộ
