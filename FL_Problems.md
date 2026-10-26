Trong các thuật toán như FedAvg, server gộp mô hình bằng cách lấy trung bình cộng hoặc chỉ dựa vào số lượng dữ liệu của từng client mà bỏ qua sự khác biệt giữa các mô hình cục bộ -> khi mà non-iid hoặc lệch số lượng : mô hình của các client sẽ đi lệch so với đám đông -> nếu vẫn gộp chung với trọng số ngang nhau sẽ kéo tụt độ chính xác của mô hình toàn cục

Để biểu diễn 1 mô hình, giả sử trên dữ liệu 2 chiều (x, y), với mặt loss được tạo ra 1 cái bạt được chống ở dưới bởi các thanh chống (bộ w) , và mặt xác suất (được tạo bằng cách đặt k cái bạt chồng lên nhau, mà tại mọi điểm x, y thì độ cao của k cái bạt cộng lại phải bằng 1 (softmax), ở đây coi k = 2 (2 tấm bạt chồng nhau), thì tức là tại 1 điểm nào đó, độ cao của bạt class 1 + độ cao của bạt class 2 phải = 1

Tầng trích xuất & phân loại: coi như là việc kéo dãn mặt sàn nhà sao cho dữ liệu của k class tự động tụ tập về k góc riêng trên sàn

Lớp linear k chiều : k nhóm thanh chống riêng cho từng class : mỗi nhóm thanh chống dùng để chống bạt tương ứng, sao cho tổng độ cao tại mỗi vị trí của các tấm bạt = 1

Mặt xác suất : Tại góc sàn chứa nhóm dữ liệu thuộc class 1 -> các thanh chống của class 1 sẽ đẩy tấm bạt xác suất tương ứng lên thật cao => độ cao tại vùng tương ứng của các tấm bạt còn lại phải thấp xuống, tương tự cho các class khác

Mặt loss : 1 mô hình tốt là 1 bộ w làm thanh chống sao cho tấm bạt mặt loss là 1 mặt cong lồi lóm, mà tại mỗi vùng dữ liệu của các class trên sàn, tấm bạt bị trũng xuống, còn các khoảng ở giữa các vùng dữ liệu tấm bạt sẽ nhô lên

Như vậy, khi train trên dữ liệu non-iid, ví dụ 1 client có 90 % là class 1 , sẽ dẫn đến sự thay đổi của k bộ chống cho k tấm bạt xác suất như sau

- Client 1 chỉ nhìn thấy mẫu của class 1 -> SGD liên tục đẩy cao thanh chống của tấm bạt class 1 -> 9 nhóm thanh chống còn lại tại cùng vùng phải thấp xuống , đồng thời các nhanh thanh chống khác với dữ liệu nằm ở vùng khác trên sàn lại không được điều chỉnh -> ko thể kéo bạt xác suất tại các vùng đó đủ cao lên được, lại còn kéo liên tục class 1 dẫn đến độ cao mặt xác suất của các vùng xung quanh cũng phải cao lên để duy trì tính liên tục -> các nhóm thanh chống khác tại vùng tương tự ko được nâng cao lên -> thấp hơn thanh chống bị ảnh hưởng ở class 1
- Với mặt loss, do chỉ tiếp xúc với class 1 -> chỉ chăm chăm điều chỉnh w sao cho tại vùng dữ liệu của class 1 trên sàn -> mặt loss trũng xuống hết cỡ mà các vùng khác bị bỏ rơi ->
- Bộ w do ko được tiếp xúc với mẫu khác, nên bộ trích xuất chỉ đi tìm kiếm các đặc trưng của class 1 mà ko đi tìm đặc trưng của các class khác
  Như vậy , khi tổng hợp lên server -> client 1 ép vùng class 1 của xác suất cao lên, client 2 thì ép vùng class 2 của mặt xác suất cao lên mà vùng class 1 lại rất thấp => trung hòa
