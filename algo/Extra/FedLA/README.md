FedLA : thuật toán tổng hợp mô hình học liên kết có nhận thức về nhãn (Label Aware)

Ý tưởng : Định giá client theo độ hiếm và độ phủ của nhãn thay vì tổng số mẫu thô. Trong FedAvg truyền thống, trọng số của 1 client chỉ phụ thuộc vào tổng số mẫu dữ liệu, nếu 1 client có rất nhiều mẫu và bị non-iid -> kéo mô hình toàn cục. Thay vì gom tất cả mẫu vào 1 rổ rồi chia tỉ lệ, FedLA chia nhỏ việc đánh giá với từng nhãn riêng : mỗi nhãn xuất hiện trong vòng giao tiếp đều có tổng giá trị đóng góp ngang nhau (=1), client nào nắm tỉ lệ lớn của 1 nhãn sẽ nhận được phần điểm tương ứng của nhãn đó (trong số tổng là 1) , sau đó mới cộng dồn điểm của tất cả các nhãn tại client đó để ra trọng số cuối cùng

Bước 1 — Tính trọng số của từng client đối với từng nhãn riêng biệt ($l_j$)<sup></sup> : Đầu tiên, server tính tổng số lượng mẫu của nhãn $l_j$ có mặt trên toàn bộ $k$ client trong vòng hiện tại<sup></sup>:

$$
S(l_j) = \sum_{i=1}^{k} S(c_i, l_j)
$$

Sau đó, trọng số của client $c_i$ đối với riêng nhãn $l_j$ được tính bằng tỉ lệ mẫu nhãn $l_j$ mà client đó sở hữu so với tổng mẫu của nhãn đó<sup></sup>:

$$
\theta(c_i, l_j) = \frac{S(c_i, l_j)}{S(l_j)}
$$

Bước này đảm bảo với một nhãn $l_j$ cụ thể, client nào có nhiều mẫu của nhãn đó hơn sẽ có tiếng nói lớn hơn đối với nhãn đó, và tổng trọng số của mọi client trên cùng một nhãn luôn bằng 1 ($\sum_{i=1}^k \theta(c_i, l_j) = 1$)<sup></sup>.

Bước 2 — Tính tổng trọng số của client trên tất cả các nhãn<sup></sup> : Server cộng dồn trọng số của client $c_i$ qua toàn bộ $n$ nhãn<sup></sup>:PDF+ 1

$$
\theta(c_i) = \sum_{j=1}^{n} \theta(c_i, l_j) = \sum_{j=1}^{n} \frac{S(c_i, l_j)}{S(l_j)}
$$

Bước này giúp các client sở hữu độ đa dạng nhãn cao (có mặt ở nhiều nhãn $l_j$) hoặc nắm giữ độc quyền các nhãn hiếm (tỉ lệ $\frac{S(c_i, l_j)}{S(l_j)}$ cao) tích lũy được điểm số $\theta(c_i)$ lớn<sup></sup>.

Bước 3 — Chuẩn hóa tuyến tính để lấy trọng số cập nhật mô hình cuối cùng<sup></sup> : Trọng số chính thức $\theta_{FedLA}(c_i)$ của client $c_i$ là tỉ lệ giữa điểm của nó so với tổng điểm của tất cả $k$ client trong vòng giao tiếp<sup></sup>:

$$
\theta_{FedLA}(c_i) = \frac{\theta(c_i)}{\sum_{x=1}^{k} \theta(c_x)}
$$

Cuối cùng, mô hình toàn cục mới được tổng hợp bằng trung bình có trọng số từ các mô hình cục bộ<sup></sup>:

$$
w^{global} = \sum_{i=1}^{k} \theta_{FedLA}(c_i) w_i
$$

Điểm yếu

1 . FedLA bị lộ phân phối nhãn tại mỗi client

2 . Nhãn phổ biến với 1000 mẫu cũng có đóng góp (quyền lực ngang bằng) như nhãn hiếm gặp 1 mẫu, từ đó phóng đại nhiễu cực đoan. Khi đó nếu client i học với 1 mẫu hiếm duy nhất nhưng vẫn được đánh trọng số cao do giữ 100% mẫu của nhãn này -> ngày càng kéo $w^{global}$ đi lệch

3. Nếu 1 client giữ 5000/x (x>>5000) mẫu A và 1/1 mẫu B trên toàn cục -> điếm số sẽ rất lớn, dù A được train trên A (đã lệch) , việc tăng trọng số $\theta_{FedLA}(c_i)$ của client này để khuếch đại mẫu B lại càng làm cho A lệch hơn
4. Trong thực nghiệm của FedLA, tác giả đã ép mọi client đều có số mẫu y hệt nhau, chỉ lệch số mẫu /nhãn , và khi non-iid rất nặng, mỗi client giữ 1 nhãn thì tất cả đều được 1 điểm, coi như lấy trung bình tất cả
