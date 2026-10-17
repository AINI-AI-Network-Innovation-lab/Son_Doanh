### Ký hiệu

- $N$: số client, $D_i$: tập dữ liệu của client $i$
- $p^{(i)}(y=j)$: phân phối dữ liệu trên client $i$ với nhãn $j$
- $\eta$: learning rate
- $F(w)$: hàm loss toàn cục, $F_i(w)$: loss tại client $i$
- $\psi_i$: trọng số của model client $i$ trong model global, dựa trên số mẫu tại mỗi client $i$
- $\tilde{\psi}_i(t)$: trọng số của model client $i$ trong global model tại vòng giao tiếp thứ $t$, thay đổi thích ứng theo vòng
- $\theta_i(t)$: góc tức thời tạo bởi vector gradient cục bộ của loss tại client ở cuối vòng giao tiếp và gradient của loss toàn cục trước khi train
- $\tilde{\theta}_i(t)$: góc làm mượt của thiết bị $i$ ở vòng $t$, bằng trung bình các góc tức thời qua các vòng huấn luyện trước đó
- $f(\tilde{\theta}_i(t))$: hàm phi tuyến định lượng mức độ đóng góp của client $i$ từ góc làm mượt
- $\alpha$: hằng số nguyên dương điều chỉnh tốc độ giảm của hàm $f$
- $\nu$ (nu), $\epsilon$ (epsilon): 2 hàm số trong hàm phi tuyến $f$, với $\nu$: ngưỡng góc nhỏ mà tại đó hàm $f$ tiến về $\alpha$, tỉ lệ thuận với $\alpha$ và $\epsilon$: giới hạn của hàm $f$ khi góc tiến đến $\frac{\pi}{2}$, tỉ lệ nghịch với $\alpha$

---

### Ý tưởng

Trong FedAvg, client được gán trọng số phụ thuộc vào số mẫu tại từng thiết bị (tốt cho IID, không tốt cho Non-IID) $\rightarrow$ FedAdp nhận diện mối liên hệ giữa phân phối nhãn trên thiết bị và mức độ đóng góp của thiết bị đó, bằng cách:

- **Đo lường đóng góp bằng góc giữa gradient:** Sau mỗi vòng giao tiếp, ở cuối vòng, đo khoảng cách giữa gradient hàm loss cục bộ và gradient của hàm loss theo model được gửi xuống ở đầu vòng giao tiếp. Nếu góc $\theta_i(t)$ nhỏ $\rightarrow$ gradient của client cùng hướng với mục tiêu chung $\rightarrow$ đóng góp tích cực và ngược lại (dùng hàm arccos để tính ra góc).
- **Làm mượt góc và lượng hóa bằng hàm phi tuyến:** Tại mỗi vòng, các góc tức thời có thể thay đổi ngẫu nhiên do dữ liệu Non-IID tại client đó làm cho gradient của loss cục bộ trỏ đi các hướng. Server sau khi nhận model cuối vòng giao tiếp của các client sẽ tính góc và góc làm mượt (bằng trung bình cộng các góc đến hiện tại) mới với công thức (1):

$$
\tilde{\theta}_i(t) = \frac{t-1}{t}\tilde{\theta}_i(t-1) + \frac{1}{t}\theta_i(t)
$$

Góc làm mượt được đưa qua hàm phi tuyến $f$ để định lượng mức đóng góp, với góc làm mượt như thế thì client này đóng góp bao nhiêu vào global từ đầu đến giờ:

$$
f(\tilde{\theta}_i(t)) = \alpha \left( 1 - e^{-e^{-\alpha(\tilde{\theta}_i(t) - 1)}} \right)
$$

Khi góc làm mượt $\tilde{\theta}_i(t)$ tăng lên (tức là thiết bị có dữ liệu càng bị lệch pha, hướng gradient càng đi lệch so với mục tiêu chung), giá trị đầu ra của hàm $f$ sẽ giảm xuống. Ngược lại, khi góc $\tilde{\theta}_i(t)$ rất nhỏ (thiết bị đóng góp tích cực, đi đúng hướng), hàm $f$ sẽ trả về giá trị cao tối đa xấp xỉ bằng $\alpha$.

---

### Quy trình

1. Client nhận được global model $w^t$, train 1 số step, thu được $w_i^t$ là model sau vòng giao tiếp $t$ tại client $i$.
2. Tính $\Delta w = w_i^t - w^t$, rồi gửi cho server.
3. Server chia cho $-\eta$ (learning rate), thu được gradient của hàm loss cục bộ: $\nabla F_i(w(t))$ (do $w_i^t = w^t - \eta \nabla F_i(w_i^t)$).
4. Tính góc tức thời $\theta_i^t$ của client $i$ với global sau vòng giao tiếp $t$ để xem độ đóng góp của client $i$ vào gradient toàn cục.
5. Tính góc làm mượt của client với công thức (1) và đưa qua hàm $f$ phi tuyến để tính mức độ đóng góp của client $i$ tại vòng lặp $t$ này, với kết quả là điểm dành cho client $i$.
6. Sau khi đi qua hàm này, kết quả được đưa qua hàm softmax để chuẩn hóa thành trọng số cho từng client.
7. Server tổng hợp ra model mới từ bộ trọng số này với công thức:

$$
w^{t+1} = w^t + \sum_{i=1}^N \tilde{\psi}_i^t \Delta w_i^t
$$

và gửi cho client ở vòng tiếp theo.
