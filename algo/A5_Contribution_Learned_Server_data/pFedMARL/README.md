pFedMARL : Học liên kết kết hợp học tăng cường đa tác tử (MARL) giải quyết đồng thời bài toán Non-IID, cá nhân hóa mô hình và chống lại các thiết bị độc hại, dựa trên kiến trúc TD3 (Twin Delayed DDPG), đa tác tử :

FedAvg : tạo ra mô hình toàn cục có accu trung bình, nhưng hoạt động kém hiệu quả tại các client non-iid

Các pp cá nhân hóa truyền thống : cố gắng cân bằng giữa toàn cục và cục bộ thông qua 1 hệ số điều chuẩn $\lambda$, nhưng lại ép dùng chung 1 hệ số tĩnh cho mọi thiết bị qua mọi giai đoạn

DRL 1 phía : đặt tác tử học tăng cường đơn lẻ ở phía Server để lọc bỏ client hại, bỏ qua bài toán cá nhân hóa mô hình cho từng client

Tác tử tại server $Agent_g$ : đóng vai trò quan sát hành vi của các client thông qua mỗi vòng giao tiếp t (quan sát trạng thái lệch pha và lịch sử cập nhật của client) để tự động điều chỉnh vector trọng số tổng hợp $a_g^t$ của các client, giúp lọc bỏ các cập nhật nhiễu hoặc tấn công từ thiết bị độc hại, bằng cách bóp trọng số của client phá hoại về gần 0

Tác tử tại client $Agent_i$ : mỗi client sở hữu 1 tác tử RL riêng để tự động điều chỉnh hệ số cá nhân hóa $a_i^t \in [0,1]$ trong từng vòng giao tiếp t nhằm cân bằng giữa việc tiếp thu tri thức toàn cục từ server (giai đoạn đầu mở hệ số lớn) và việc tinh chỉnh mô hình theo dữ liệu cục bộ (giai đoạn sau mở hệ số nhỏ lại)

B1 : Thiết lập không gian quan sát

Quan sát của server ($o_g^t$) tại vòng t : server thu thập thông tin để đưa vào tác tử gồm các vector :

$$
o_g^{\langle t \rangle} = \left(l^{\langle t \rangle}, \; c^{\langle t \rangle}, \; d^{\langle t \rangle}, \; n^{\langle t \rangle}, \; t\right)
$$

$l^t$ : vector chứa giá trị hàm mất mát kiểm định cục bộ $\mathcal L (\theta_i^t, D_{validate, i})$ của từng client

$c^t$ : vector độ tương đồng cosine hướng cập nhật giữa client i và server mỗi vòng giúp server phát hiện client đi ngược đám đông

$d^t$ : (khoảng cách $l_2$) : vector khoảng cách chuẩn euclide giữa mô hình toàn cục và client i : giúp phát hiện các đột biến biên độ do tấn công bơm nhiễu

$n^t$ : số lượng mini-batch dữ liệu của mỗi client

t : chỉ số vòng hiện tại

Quan sát của tác tử tại client thứ i ($o_i^t$) : gồm các giá trị đơn lẻ như $c_i^t$, $d_i^t$, $t$ , kết hợp với đánh giá hiệu năng (sai số MSE và F1-Score) trên tập kiểm định với mô hình cục bộ và mô hình toàn cục

B2 : Không gian hành động và quy trình cập nhật

1 > Hành động của server & tổng hợp mô hình toàn cục : Tác tử server xuất ra vectpr hành động thô qua mạng chính sách $\pi_g$ kết hợp nhiễu khám phá $\epsilon$, sau đó đưa qua hàm softmax để tạo thành vector trọng số tổng hợp client chuẩn hóa (tổng bằng 1)

$$
a_g^{\langle t \rangle} = \text{Softmax}\left(\pi_g(o_g^{\langle t \rangle}) + \epsilon\right)
$$

Mô hình toàn cục tại vòng giao tiếp t được cập nhật với vector trọng số vừa tổng hợp được 

$$
\theta_g^{\langle t \rangle} = \sum_{i=1}^{M} a_{g,i}^{\langle t \rangle} \theta_i^{\langle t-1 \rangle}
$$

2 > Tác tử tại client i xuất ra hệ số cá nhân hóa $\alpha_i^t \in [0, 1]$, hệ số này điều tiết mức độ ràng buộc khoảng cách trong hàm mất mát huấn luyện cục bộ

$$
\mathcal{L}_i^{\text{pFedMARL}} = \frac{1}{\vert{}\mathcal{D}_i\vert{}} \sum_{(x,y) \in \mathcal{D}_i} l\left(f_{\theta_i}(x), y\right) + \frac{a_i^{\langle t \rangle}}{2} \left\Vert{}\theta_i - \theta_g^{\langle t \rangle}\right\Vert{}_2^2
$$

Ý nghĩa : thay vì giữ hệ số $\lambda$ tĩnh cho khoảng cách $l_2$, tại khoảng thời gian đầu, $\alpha_i^t$ lớn, làm cho Loss tăng cao, từ đó $\theta_i$ gần với $\theta_g^t$ hơn để giảm loss , tức là phải học nhiều hơn từ mô hình toàn cục và ngược lại khi $\alpha$ thấp cho phép mô hình cục bộ đi xa hơn khỏi mô hình toàn cục . $\frac{1}{\vert{}\mathcal{D}_i\vert{}} \sum_{(x,y) \in \mathcal{D}_i} l\left(f_{\theta_i}(x), y\right)$ là trung bình loss trên toàn bộ mẫu tại client i

B3 : Hàm phần thưởng hợp tác (Reward Function)

Để định hướng cho các mạng TD3, phần thưởng $r^t$ được tính bằng logarit âm của hàm mất mát kiểm định sau mỗi bước cập nhật

$$
r^{\langle t \rangle} = -\log \mathcal{L}\left(\theta^{\langle t \rangle}; \mathcal{D}_{val}\right)
$$

Phạt nặng các hành động làm giảm hiệu năng mô hình (nếu loss tăng cao do hành động sai -> log dương lớn -> r âm lớn và ngược lại)

Cơ chế bù đắp chênh lệch biên độ giữa giai đoạn đầu và cuối

Giai đoạn đầu huấn luyện, loss tụt rất nhanh và ngược lại khi đến giai đoạn cuối huấn luyện, do đó nếu dùng hàm tuyến tính, tác tử chỉ ưu tiên tối ưu hóa giai đoạn đầu để Loss tụt rất nhanh nhưng không cảm nhận được ở giai đoạn cuối vì biên độ giảm quá nhỏ.

Độ chênh lệch logarit từ -log(5) -> -log(2.5) = 0.70 , -log(0.05) -> -log(0.02) = 0.92

Do đó qua hàm logarit, biên độ thay đổi loss được phóng to ra
