Vấn đề : với dữ liệu non-iid, mô hình local sau khi train trên tập dữ liệu lệch sẽ drift khỏi tối ưu toàn cục, khi mà mỗi client chuyên về 1 mảng, nhưng khi tổng hợp thì lại bị lơ lửng ở giữa , gây ra nhận diện kém. FedProx và Scaffold sửa drift bằng cách ép cho $w_k$ gần với $w^t$, hoặc hiệu chỉnh với control variate $c$. 

MOON sửa drift ở không gian biểu diễn : đầu ra của các lớp ẩn trong mạng neuron, trước khi đi vào lớp phân loại cuối cùng

- Bài toán : client $k$ giữ tập dữ liệu $\mathcal{D}^k$, mục tiêu tối ưu

$$
\min_{w}\ \mathcal{L}(w)=\sum_{k=1}^{N}\frac{|\mathcal{D}^k|}{|\mathcal{D}|}\,L_k(w),\qquad L_k(w)=\mathbb{E}_{(x,y)\sim\mathcal{D}^k}\big[\ell_k(w;(x,y))\big]
$$

Với $L_k(w)$ là hàm loss cục bộ, $\ell_k$ là loss trên từng mẫu cục bộ, và $\mathcal{L}(w)$ là hàm loss toàn cục

Thông thường, các mạng tích chập có cấu tạo : 

$$
\text{Ảnh đầu vào} \to \text{Khối Tích chập (CNN)} \to \text{Làm phẳng (Flatten)} \to \text{Các lớp Fully-Connected (FC / NN)} \to \text{Lớp Output (Logits)}
$$

Trong MOON, có thêm 1 lớp MLP 2 tầng nằm sau lớp kích hoạt cuối cùng và trước lớp tuyến tính cuối cùng, trước khi đưa qua softmax để phân loại, với MLP 2 tầng gồm : 2 lớp tuyến tính và 1 hàm kích hoạt ở giữa

$$
\text{Ảnh } x \to \underbrace{\text{Tích chập} \to \text{Làm phẳng} \to \text{Các lớp FC ban đầu}}_{\textbf{Base Encoder } f(\cdot)} \xrightarrow{\text{vector } h} \underbrace{\text{MLP 2 tầng}}_{\textbf{Projection Head } g(\cdot)} \xrightarrow{\textbf{vector } z} \underbrace{\text{Lớp FC cuối cùng}}_{\textbf{Output Layer}} \to \hat{y}
$$

Với lớp FC cuối cùng chính là lớp tuyến tính & softmax để phân loại, vector đầu ra tại MLP 2 tầng được đưa vào đồng thời lớp phân loại (tuyến tính + softmax) , và dùng để đo $L_{\text{con}}$ : đo góc giữa dự đoán đầu ra tại thời điểm hiện tại với dự đoán đầu ra đầu vòng giao tiếp

```text
                        L_sup (Phân loại ảnh) ← Output Layer <--- 
                                                                │
Input x ─> [ Base Encoder ] ─(vector h)─> [ MLP 2 tầng ] ─> (vector z) ─> L_con (Đo góc với Global model)
```

Với $L_{\text{sup}}$ : hàm mất mát học có giám sát (loss giữa dự đoán và nhãn thực tế - thường là cross entropy)

Với $L_{\text{con}}$ (Model-constrastive loss - hàm mất mát cấp độ mô hình) : Thay vì so sánh 2 bức ảnh khác nhau (SimCLR), $L_{\text{con}}$ lấy cùng 1 bức ảnh, đưa qua 3 mô hình :

- $z$ : đầu ra của mô hình local hiện tại
- $z_{\text{glob}}$ : đầu ra của mô hình global gửi xuống tại đầu vòng giao tiếp
- $z_{\text{prev}}$ : đầu ra của mô hình local tại cuối vòng giao tiếp trước

Tức là tại mỗi bước lặp, phải lưu trữ 3 model : $w^{t-1}$, $w^t$, $w_{\text{current}}$, sinh đồng thời ra 3 vector $z$ như trên, tính $L_{\text{con}}$ giữa 3 vector $z$ này với công thức : hàm tính độ tương đồng giữa 2 vector cosine

$\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ và nhiệt độ $\tau$ 

$$
L_{\text{con}} = -\log \frac{\exp(\text{sim}(z, z_{\text{glob}}) / \tau)}{\exp(\text{sim}(z, z_{\text{glob}}) / \tau) + \exp(\text{sim}(z, z_{\text{prev}}) / \tau)}
$$

**Tử số:** Đo mức độ đồng thuận giữa $z$ và $z_{\text{glob}}$. Muốn $L_{\text{con}}$ nhỏ thì $z$ phải kéo sát lại gần $z_{\text{glob}}$ (học theo cái nhìn chuẩn của toàn cầu).  

**Mẫu số:** Có thêm số hạng $\text{sim}(z, z_{\text{prev}})$. Việc tối thiểu hóa loss sẽ tự động **đẩy** $z$ **ra xa** $z_{\text{prev}}$, ngăn mô hình lặp lại các sai lệch quá khớp cục bộ ở vòng trước.

Nhiệt độ $\tau$ : khi $\tau$ nhỏ, $\frac{\text{sim}}{\tau}$ lớn, từ đó khuếch đại sự bất tương đồng giữa 2 vector trong hàm $\text{sim}(z_1, z_2)$ , khuếch đại lần 2 thông qua hàm $\exp$

- Vậy tại sao ko cắm thẳng vector $h$ vào $L_{\text{con}}$ và $L_{\text{sup}}$ mà phải nhét thêm 1 lớp MLP 2 tầng vào giữa : đây coi như là lớp đệm, nếu cắm thẳng $h$ vào $\rightarrow$ khi gradient dội ngược về, bắt buộc $h$ phải thay đổi cho cả $L_{\text{sup}}$ (yêu cầu phải giữ được các đặc trưng) và $L_{\text{con}}$ (gọt bớt đặc trưng để 2 vector về cùng hướng) , khi thêm 1 lớp MLP 2 tầng : phần lớn trách nhiệm điều chỉnh là của MLP, lớp Base encoder đằng trước $h$ vẫn có thể giữ nguyên

Tại sao phải ép $z$ về gần $z_{\text{glob}}$ và ép $z$ ra xa $z_{\text{prev}}$ : 

- Ép $z \rightarrow z_{\text{glob}}$ kéo không gian biểu diễn của client về chuẩn mực chung, không để mô hình cục bộ đi vào các cực tiểu cục bộ
- Ép $z$ ra xa $z_{\text{prev}}$ : mô hình của client ở vòng trước đã quá khớp với dữ liệu cục bộ và có biểu diễn bị lệch, do đó nếu chỉ ép $z \rightarrow z_{\text{glob}}$ mà ko đẩy ra xa $z_{\text{prev}}$, mô hình có xu hướng giữ nguyên như vòng trước

Ý nghĩa của điều chỉnh không gian biểu diễn : mỗi bức ảnh về 1 vật thể đều được embedd thành 1 vector $k$ chiều trong không gian biểu diễn, do đó MOON ép các bức ảnh nhận diện cùng vật thể được biểu diễn thành các vector gần nhau trong không gian $k$ chiều , tạo ra sự phân cụm trong không gian, mà không cần ép $w_k$ gần với $w$

- Tác dụng của MLP:
    - Nếu không có MLP, mẫu $x$ cùng là class A , nhưng tại client 1, class A được phân cụm ở vùng 1 trong không gian, client 2 , class A được phân cụm ở vùng 2 trong không gian, khi tổng hợp toàn cục , 2 cụm này nằm xa nhau trong không gian biểu diễn
    - Nếu có MLP,với model được phân phối tại đầu vòng giao tiếp và mẫu $x$ cùng là class A, tại client 1, khi huấn luyện nhận diện class A, MLP sẽ làm cho các mẫu thuộc class A sau khi đi qua Base encoder (vẫn là 1 vector không thuộc cụm chuẩn hóa trong không gian) được dịch chuyển phân cụm sao cho gần với model toàn cục gửi ở đầu vòng nhất có thể, tương tự với client 2. Như vậy, khi tổng hợp toàn cục, cụm class A được thống nhất giữa các client ko làm thay đổi vị trí trong không gian $k$ chiều vector hóa

$\Rightarrow$ Server tổng hợp lại với thuật toán FedAvg tại server

- Ưu điểm : MOON cho phép các bộ trọng số $w_k$ tại client được trôi tự do mà không ràng buộc vào $w_{\text{global}}$, hội tụ nhanh , tiết kiệm băng thông
- Nhược điểm : Tốn kém bộ nhớ tại client (do phải chạy 3 model 1 lúc) , tăng thười gian tính toán cục bộ, nhạy cảm với nhiệt độ $\tau$