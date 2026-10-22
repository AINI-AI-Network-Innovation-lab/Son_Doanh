Ý tưởng tổng quát

FedAvg và các phương pháp cải tiến trọng số trước đó : hoặc yêu cầu giả định mạnh (iid, proxy data. hoặc thu thập phân phối nhãn), hoặc dựa trên số nhãn làm tín hiệu

Thay vì ép buộc client trong quá trình học hoặc tính toán độ tương quan trên server, FedAAW đặt ra câu hỏi : làm sao để xác định trọng số client $p_k$ sao cho cận trên sai số hội tụ của mô hình toàn cục là thấp nhất có thể

Bình phương chuẩn đạo hàm toàn phần tính trước bước cập nhật cục bộ $\Vert{}\nabla F_k(w_{t,0}^k)\Vert{}^2$ (bình phương chuẩn gradient ngay khi nhận được model từ server) có thể phản ánh độ lệch dữ liệu

Khi vừa nhận được model từ server, tính độ dốc của mình trên toàn bộ dữ liệu trước khi chạy SGD

Nếu $\Vert{}\nabla F_k(w_{t,0}^k)\Vert{}^2$ nhỏ : độ dốc trên dữ liệu cục bộ hiện tại nhỏ, mô hình hiện tại khớp tốt với dữ liệu cục bộ, quá trình huấn luyện tiếp theo sẽ ổn định và ít gây trượt -> cho trọng số lớn (do ko bị kéo đi lệch, tự tin cho trọng số cao) và ngược lại

Cách sinh trọng số cho client từ cái gradient ban đầu

Bổ đề 1 : Tổng độ lệch đi của client trong suốt các bước từ 0 đến $\tau$ bị chặn trên bởi bình phương chuẩn đạo hàm ban đầu

$$
\left\Vert{}\sum_{j=0}^{\tau-1} \nabla F_k(w_{t,j}^k)\right\Vert{}^2 \le \frac{2(\tau-1)(2\tau-1)\tau^2 L^2 \sigma_k^2 \eta^2}{3(1-2\tau(\tau-1)L^2\eta^2)} + \frac{2\tau^2}{1-2\tau(\tau-1)L^2\eta^2}\Vert{}\nabla F_k(w_{t,0}^k)\Vert{}^2
$$

Dựa trên bổ đề 1 , thiết lập được cận trên hội tụ toàn cục qua định lý 1 :

Sai số hội tụ của mô hình toàn cục (mức độ xa cách điểm tối ưu mô hình toàn cục sau T vòng giao tiếp, kí hiệu $\min_{t} \mathbb{E}\Vert{}\nabla F(w_t)\Vert{}^{2}$) lớn hay nhỏ là do tổng các gradient ban đầu của các client gánh chịu, muốn mô hình hội tụ tốt nhất, ta phải xác định $p_k$ sao cho tổng này nhỏ nhất

$$
\min_{t} \mathbb{E}\Vert{}\nabla F(w_t)\Vert{}^2 \le \frac{2(F_0 - F_*)}{T\eta} + M \sum_{k=1}^{K} \sigma_k^2 p_k + \frac{1}{T}\sum_{t=0}^{T-1}\sum_{k=1}^{K} Q_k \Vert{}\nabla F_k(w_{t,0}^k)\Vert{}^2
$$

với $Q_k = 2K \left(\frac{\vert{}\mathcal{D}\_k\vert{}}{\vert{}\mathcal{D}\vert{}} - p\_k\right)^2 + (2\tau + 2 + H)p\_k$

Mục tiêu : Tối thiểu hóa vế phải, tức là cận trên của sai số hội tụ, từ đó làm cho sai số hội tụ giảm xuống -> hàm mục tiêu bây giờ là vế phải. Bài toán trở thành : tìm bộ trọng số $p_k$ sao cho cận trên đạt giá trị cực tiểu, với điều kiện $\sum p_k =1$ , đây là bài toán tối ưu phi tuyến có ràng buộc, có thể giải bằng pp nhân tử Lagrange và pp KKT (Nhập môn các pp tối ưu) để giải ra nghiệm lý thuyết cho $\{p_k\}$ , với công thức nghiệm sau :

$$
p_k = \frac{\vert{}\mathcal{D}_k\vert{}}{\vert{}\mathcal{D}\vert{}} + a_k \frac{1}{R_k^T} - b
$$

Trong đó $R_k^T = \frac{1}{T}\sum_{t=0}^{T-1} \Vert{}\nabla F_k(w_{t,0}^k)\Vert{}^2$ là trung bình cộng độ dốc tính xuyên suốt qua toàn bộ T vòng huấn luyện -> vấn đề : trong khi train ở 1 vòng t bất kì thì ko biết trước được $R_t^k$ do phải chạy hết mới tính được, do đó công thức này mang tính định hướng chứ chưa dùng được ngay

Để giải quyết vấn đề trên, thực hiện 2 pp thay thế
Xây dựng Numerical tracker ($R_k^t$) thay vì đợi chạy hết T vòng, tác giả thiết kế 1 biến tính trung bình cộng dồn tích lũy từ vòng 0 đến t hiện tại

Chuẩn hóa qua softmax : thay thế các hằng số $\alpha_k$, $b$ thành các tham số thực nghiệm $\beta$, $\gamma$ và đưa kết quả $p_k$ cuối cùng qua softmax để chuẩn hóa trong khoảng [0,1]

Từ đó cho ra được 1 bộ trọng số $\{p_k\}$ cho các client
