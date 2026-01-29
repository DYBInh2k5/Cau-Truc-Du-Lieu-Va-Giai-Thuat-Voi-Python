# Cau-Truc-Du-Lieu-Va-Giai-Thuat-Voi-Python
Cấu trúc dữ liệu và giải thuật với python

Tổng quan về Cấu trúc dữ liệu và Giải thuật (DSA) với Python
Dự án này tập trung vào việc triển khai và phân tích các cấu trúc dữ liệu cũng như giải thuật nền tảng bằng ngôn ngữ lập trình Python.
1. Phân tích độ phức tạp (Big-O Cheat Sheet)
Việc hiểu rõ độ phức tạp là điều kiện tiên quyết để đánh giá hiệu suất của thuật toán. Dưới đây là bảng tóm tắt độ phức tạp thời gian của các cấu trúc dữ liệu và giải thuật phổ biến:
Cấu trúc dữ liệu (Data Structures)
Cấu trúc dữ liệu
Trung bình (Insert/Delete/Search)
Xấu nhất (Insert/Delete/Search)
Mảng/Stack/Queue
O(1)/O(1)/O(n)
O(1)/O(1)/O(n)
Linked List
O(1)/O(1)/O(n)
O(1)/O(1)/O(n)
Hash Table
O(1)/O(1)/O(1)
O(n)/O(n)/O(n)
Binary Search Tree
O(logn)/O(logn)/O(logn)
O(n)/O(n)/O(n)
Thuật toán sắp xếp (Sorting Algorithms)
Thuật toán
Tốt nhất
Trung bình
Xấu nhất
Bubble / Insertion Sort
O(n)
O(n 
2
 )
O(n 
2
 )
Merge Sort
O(nlogn)
O(nlogn)
O(nlogn)
Quick Sort
O(nlogn)
O(nlogn)
O(n 
2
 )

--------------------------------------------------------------------------------
2. Các cấu trúc dữ liệu quan trọng
Danh sách liên kết (Linked Lists)
• Singly Linked List: Gồm các nút (node) chứa dữ liệu và một con trỏ next trỏ đến nút tiếp theo. Nút cuối cùng sẽ trỏ đến None.
• Doubly Linked List: Mỗi nút giữ cả con trỏ prev và next, cho phép thực hiện các thao tác chèn/xóa tại vị trí bất kỳ với thời gian O(1).
Ngăn xếp và Hàng đợi (Stack & Queue)
• Stack (LIFO): Hoạt động theo nguyên lý "Vào sau, Ra trước". Ứng dụng phổ biến trong chức năng Undo/Redo, xử lý lời gọi hàm (Call Stack).
• Queue (FIFO): Hoạt động theo nguyên lý "Vào trước, Ra trước". Ứng dụng trong hệ thống gửi email marketing hàng loạt hoặc quản lý hàng đợi in ấn.
Cây và Bảng băm (Trees & Hash Tables)
• Binary Search Tree (BST): Cây nhị phân mà giá trị cây con bên trái luôn nhỏ hơn nút gốc và bên phải luôn lớn hơn.
• Hash Table: Sử dụng hàm băm (hash function) để ánh xạ khóa (key) thành chỉ số trong mảng, giúp tìm kiếm gần như tức thời (O(1) trung bình). Các xung đột (collisions) được xử lý bằng phương pháp dò tuyến tính (linear probing) hoặc kết hợp dây chuyền (chaining).

--------------------------------------------------------------------------------
3. Các giải thuật nền tảng
Đệ quy (Recursion)
Sử dụng hàm tự gọi lại chính nó để giải quyết các bài toán có cấu trúc lặp lại như Giai thừa (Factorial), dãy Fibonacci hoặc duyệt hệ thống tệp tin.
Quy hoạch động (Dynamic Programming)
Kỹ thuật tối ưu hóa bằng cách chia nhỏ bài toán thành các bài toán con chồng lấn và lưu trữ kết quả để tái sử dụng. Các bài toán tiêu biểu bao gồm:
• Nhân chuỗi ma trận (Matrix Chain-Product).
• Tìm chuỗi con chung dài nhất (Longest Common Subsequence - LCS).
Giải thuật Đồ thị (Graph Algorithms)
• Duyệt đồ thị: Depth-First Search (DFS) cho phép khám phá sâu các nhánh; Breadth-First Search (BFS) giúp tìm đường đi ngắn nhất về số cạnh.
• Đường đi ngắn nhất: Thuật toán Dijkstra tìm đường đi ngắn nhất trên đồ thị có trọng số không âm.
• Cây bao trùm tối thiểu (MST): Các thuật toán Prim-Jarník và Kruskal giúp kết nối tất cả các đỉnh với tổng chi phí thấp nhất.

--------------------------------------------------------------------------------
4. Tại sao chọn Python để học DSA?
• Tính trừu tượng cao: Giúp tập trung vào ý tưởng và bản chất thuật toán thay vì các chi tiết quản lý bộ nhớ phức tạp.
• Cú pháp dễ hiểu: Gần với ngôn ngữ tự nhiên, giúp lập trình viên phác thảo thuật toán nhanh chóng.
• Lưu ý: Python có thể chậm hơn C++ cho các tác vụ thời gian thực hoặc cực lớn, nhưng sức mạnh thực sự nằm ở việc kết hợp logic Python với các thư viện hiệu năng cao viết bằng C (như NumPy, Pandas).

--------------------------------------------------------------------------------
5. Danh sách bài tập thực hành
Hệ thống bài tập trải dài từ cơ bản đến nâng cao bao gồm:
1. Cơ bản: Kiểm tra số học, thao tác chuỗi (palindrome), tính chu vi/diện tích.
2. DSA: Tự triển khai Stack/Queue từ mảng/danh sách liên kết, thực hiện thao tác xóa nút trùng lặp, hoán đổi nút hoặc đảo ngược danh sách.
3. Dự án: Xây dựng trình soạn thảo văn bản đơn giản, mô phỏng hệ sinh thái hoặc bộ lọc âm thanh.

--------------------------------------------------------------------------------
Thông tin được tổng hợp từ các tài liệu về lập trình Python, giáo trình DSA của Michael T. Goodrich và các nguồn thảo luận cộng đồng uy tín.
