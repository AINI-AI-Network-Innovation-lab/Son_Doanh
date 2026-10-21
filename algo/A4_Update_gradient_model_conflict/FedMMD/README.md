Như phân tích trong @Vấn đề của FL, mạng CNN trích xuất đặc trưng truyền thống có tầm nhìn hẹp, nên nếu 1 client có 90% dữ liệu thuộc 1 class, các lớp trích xuất trong CNN sẽ điều chỉnh để chỉ nhận diện các đặc trưng của class đó mà quên đi các class khác

FedMMD sử dụng mạng DCMT tại client : Dilated Convolution + Transformer

DC : tích chập giãn nở : mở rộng vùng nhìn trên bức ảnh mà k làm mất độ phân giải

Transformer (Multi head attention) : Mô hình học được mối liên hệ toàn cục giữa các phần xa nhau của bức ảnh và tập trung vào các đặc trưng cốt lõi

Nhiệm vụ của mạng trích xuất : tại 1 client A, với mỗi bức ảnh và mỗi đặc trưng trong nó, đặc trưng được vector hóa thành 1 vector k chiều, được biểu diễn ở 1 vị trí trong không gian biểu diễn k chiều

DCMT chuẩn hóa việc biểu diễn các đặc trưng được trích xuất nếu giống nhau tại các client về cùng 1 vùng trước khi phân loại -> như vậy , tại client 1 đặc trưng X nằm ở đâu thì client 2 cũng thế

Tại server : Dù dùng DCMT, các client bị non-iid nặng hoặc có quá ít dữ liệu vẫn sẽ làm cho các nhóm thanh đỡ w của các tấm bạt xác suất lệch đi (tồn tại các vùng xung quanh bị ảnh hưởng do kéo 1 vùng quá nhiều - làm tấm bạt xác suất của class 1 cao lên ở mọi vùng dữ liệu , làm cho tại vị trí tương đương, các tấm bạt xác suất khác phải thấp xuống, mặc dù đó là vùng của class khác-> hiện tưởng ảo giác xác suất (ngoài ra còn bị khuếch đại bởi hàm softmax)

Server dùng các kĩ thuật

MMD : phát hiện xem client đang có xu hướng kéo cao tấm bạt xác suất nào đó lên rất cao và thả nổi các vùng bạt khác cao theo -> phân phối trọng số khác biệt so với phần còn lại : Chỉ số độ lệch MMD của client đó sẽ vọt lên rất cao

Thuật toán MMD (Maximum Mean Discrepancy) : chiếu các bộ tham số $w^k$ lên không gian chiều cao (RKHS) để đo khoảng cách phân phối giữa từng cặp mô hình client với nhau

Đám mây trọng số : Tập hợp tất cả các thành phần trong bộ trọng số w của 1 client khi biểu diễn lên đồ thị (hàng triệu chấm trên đồ thị tạo thành 1 đám điểm) , ở client iid bình thường : sẽ cho ra đám mây hình quả chuông Gauss, hầu hết xung quanh mức 0 (pp chuẩn). Ở client bị Non-iid nặng (90% là class 1) -> sẽ có các số trong bộ trọng số vọt lên rất cao , các số tụt xuống rất thấp (có thể âm)-> khi biểu diễn lên trục tọa độ -> đám điểm bị kéo dẹt, xé lẻ hoặc văng ra 2 đầu

Khi đó MMD hoạt động như sau : Nhìn đám chấm các số trong w của client 1 (đám mây 1), và client 2 (đám mây 2) -> nếu 2 đám mây có hình dáng giống nhau (cùng phân phối) -> có mức đồng thuận cao (thay vì chỉ so sánh trung bình -mean để xem độ tương đồng, thì so sánh cả độ tụ, độ rộng và vị trí)

Để có được độ tụ, độ rộng và vị trí, cần hàm nhân so sánh độ tương đồng giữa 2 số trọng số x và x'

$$
k(x, x^{'})= e^ {-\frac{\vert \vert x-x^{'} \vert \vert ^2}{2 \sigma ^2}}
$$

Để tính độ tương đồng nội bộ trong client A : $\sum_{i \neq j} k(x_i, x_j)$ với $x_i$ , $x_j$ là các số trong cùng client model, nếu x_i khác x_j nhiều -> hiệu lớn -> $k(x_i, x_j)$ rất nhỏ và ngược lại

Để tính độ tương đồng chéo giữa 2 client : 

$$
\sum k(x_i, y_j)
$$

Nếu điểm tương đồng của 2 client cao -> 2 client đang có xu hướng chồng lấn nhau nhiều không -> nếu điểm cao => cùng phân phối,  vị trí, độ rộng, độ tụ , v.v... , 2 client có xu hướng học giống nhau, từ đó có được khoảng cách giữa mỗi client đến tất cả client còn lại (tổng độ lệch đến tất cả client còn lại = chỉ số độ lệch cho client i)

Sắp xếp các bộ trọng số theo chỉ số độ lệch của các client từ nhỏ đến lớn, các client có chỉ số độ lệch ở mức trung bình thì giữ lại, nếu chỉ số độ lệch rất cao -> đánh dấu đáng ngờ $W_{s1}$, $W_{s2}$,...

Để không bị nhầm lẫn cảm tính, kiểm định 

$$
SKNQ = \frac{\vert W_{s1} - W_{s2} \vert}{W_{\max} - W_{\min}}
$$

Với W_max là mô hình có chỉ số độ lệch cao nhất, W_min là thấp nhất, $\vert W_{s1} - W_{s2} \vert$ : khoảng cách giữa các chỉ số độ lệch đáng ngờ, nếu SKNQ > 0.5 -> loại bỏ (hiệu quả nếu W_s1 và s2 tương phản nhau, chứ nếu cùng có chỉ số độ lệch cao nhưng độ tương đồng chéo lại tương đương thì k giải quyết được
