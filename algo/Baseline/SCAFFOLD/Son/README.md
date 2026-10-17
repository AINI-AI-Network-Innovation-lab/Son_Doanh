- Ý tưởng : nếu server liên lạc được với mọi client ở mọi bước, bước lí tưởng của client là đi ngược chiều gradient của loss toàn cục
    
    $$
    w\leftarrow w-\lambda\cdot\frac1N\sum_{j=1}^N\nabla F_j(w)
    $$
    
    Nhưng để có gradient toàn cục, thì phải có gradient cục bộ của tất cả client tại bước đó → cần giữ liên kết liên tục (tốn kém) , và cập nhật này giống hệt như chạy SGD trên dữ liệu IID → không có client drift 
    
- SCAFFOLD xấp xỉ gradient toàn cục

$$
\frac1N\sum_{j=1}^N\nabla F_j(w)
$$

bằng 2 ước lượng :

- c_k (control variate của client k) : ước lượng hướng gradient của riêng client k tại đầu vòng giao tiếp thứ t
    
    $$
    (c_k \approx \nabla F_k(w^t))
    $$
    
- c (control variate của server) : ước lượng hướng gradient toàn cục tại vòng giao tiếp thứ t : $c \approx \nabla f(w^t)$
    
     $c\approx\frac1N\sum_jc_j$
    

2 đại lượng này là vector cùng chiều với w, khởi tạo c_k = c = 0, giữ $c=\frac1N\sum_kc_k$                                              

$$
w_k^{t, s+1} = w_k^{t, s} - \lambda \Big[ g_k(w_k^{t, s}) - c_k + c \Big] = w_k^{t, s} \underbrace{-\; \lambda g_k(w_k^{t, s})}_{\text{Bước SGD cục bộ}} + \underbrace{\lambda c_k}_{\text{Cộng thêm } c_k} \underbrace{-\; \lambda c}_{\text{Trừ đi } c}
$$

Khi bốc ngẫu nhiên 1 mini-batch ở bước s, gradient tức thời $g_k(w_k^{t,s})$ được cấu tạo từ 2 thành phần : 

$$
g_k(w_k^{t,s}) = \underbrace{\nabla F_k(w_k^{t,s})}_{\text{Xu hướng chung của toàn mẫu client } k} + \underbrace{\epsilon^{t,s}}_{\text{Nhiễu ngẫu nhiên của riêng mini-batch đó}}
$$

Với phần $\nabla F_k(w_k^{t,s}) = \frac{1}{n_k} \sum_{i=1}^{n_k} \nabla f_i(w_k^{t,s})$ là xu hướng chung của  toàn mẫu theo $w_k^{t,s}$ : kéo $w_k^{t,s}$ về phía cực tiểu gần nhất của toàn mẫu tại client k 

$\epsilon ^{t, s}$ : biến thiên hỗn loạn do mini-batch gây ra quanh xu hướng chung, kéo $w_k$ lệch đi các hướng nhỏ xung quanh hướng của $\nabla F_k$

Tại bước đầu ở xuất phát $w^t$, có $\nabla F_k(w_k^{t,s}) \approx c_k$ do đó $g_k(w_t^{t,s}) \approx c_k + \epsilon ^{t,s}$  

Khi +$\lambda c_k$ , nó sẽ triệt tiêu bớt đi lực kéo về cực tiểu gần nhất của toàn mẫu tại client k (bớt chứ không hết, và lực kéo sai lệch dần khi chạy đến các step sau khi mà $w_k^{t,s}$ thay đổi dần theo step s) 

Khi -$\lambda c$, w bị kéo về phía cực tiểu gần nhất trên bề mặt loss toàn cục f thay vì bị giữ ở cực tiểu gần nhất trên bề mặt loss của client k

- 2 phương án tạo và lưu trữ c_k (ước lượng độ dốc cục bộ tại đầu vòng = trung bình gradient trên toàn mẫu) tại client k :
    - Tại cuối mỗi vòng giao tiếp t : thực hiện tính gradient của loss trung bình trên toàn mẫu ⇒ tốn kém :
    
    $$
    c_k^+ = \frac{1}{\vert{}D_k\vert{}} \sum_{i \in D_k} \nabla f_i(w^t)
    $$
    
    - Do c_k là trung bình gradient trên toàn mẫu , nên có thể tính thông qua tổng trung bình gradient của từng batch
    
    $$
    c_k^+ \approx \frac{1}{K} \sum_{s=0}^{K-1} g_k(w_k^{t, s})
    $$
    

Với $c_k^+$ là $c_k$ cho vòng giao tiếp tiếp theo

Từ công thức cộng dồn tổng độ dời từ w cuối cùng trong vòng giao tiếp với w_t được phát ở đầu vòng

$w_k^{t, K} - w^t = -\lambda \sum_{s=0}^{K-1} g_k(w_k^{t, s}) + K\lambda(c_k - c)$

Chuyển số hạng chứa tổng gradient sang vế trái, và chuyển $(w_k^{t, K} - w^t)$ sang vế phải:

$\lambda \sum_{s=0}^{K-1} g_k(w_k^{t, s}) = K\lambda(c_k - c) - (w_k^{t, K} - w^t)$

Chia cả hai vế cho $K\lambda$:

$\frac{1}{K} \sum_{s=0}^{K-1} g_k(w_k^{t, s}) = (c_k - c) + \frac{1}{K\lambda}(w^t - w_k^{t, K})$

- Ưu điểm : giữ cho w cục bộ không bị rơi vào cực tiểu cục bộ, sau nhiều bước lặp, độ dốc $\nabla F_k(w_k^{t,s})$ giảm dần , khi đó c_k >> độ dốc hiện tại → kéo w ra hoàn toàn khỏi cực tiểu cục bộ
- Nhược điểm : tốn kém chi phí truyền thông