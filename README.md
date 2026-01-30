# Cấu Trúc Dữ Liệu và Giải Thuật với Python

## 📚 Tổng Quan

Chào mừng bạn đến với hướng dẫn toàn diện về Cấu trúc Dữ liệu và Giải thuật (Data Structures & Algorithms - DSA) sử dụng Python. Đây là nền tảng quan trọng cho mọi lập trình viên muốn nâng cao kỹ năng giải quyết vấn đề và tối ưu hóa code.

## 🎯 Tại Sao Học DSA?

- **Tối ưu hóa hiệu suất**: Giúp viết code chạy nhanh hơn và tiết kiệm bộ nhớ
- **Phỏng vấn kỹ thuật**: Hầu hết các công ty công nghệ đều kiểm tra kiến thức DSA
- **Tư duy logic**: Rèn luyện khả năng phân tích và giải quyết vấn đề
- **Nền tảng vững chắc**: Cơ sở để học các công nghệ cao cấp hơn

## 🐍 Tại Sao Chọn Python?

- **Cú pháp đơn giản**: Dễ đọc, dễ viết, tập trung vào logic
- **Thư viện phong phú**: Hỗ trợ sẵn nhiều cấu trúc dữ liệu
- **Phổ biến**: Được sử dụng rộng rãi trong công nghiệp và giáo dục
- **Prototype nhanh**: Kiểm tra ý tưởng và thuật toán dễ dàng

---

## 📊 Độ Phức Tạp Thuật Toán

### Time Complexity (Độ Phức Tạp Thời Gian)
Đo lường thời gian thực thi của thuật toán theo kích thước input.

| Ký hiệu | Tên | Ví dụ |
|---------|-----|-------|
| O(1) | Constant | Truy cập mảng theo index |
| O(log n) | Logarithmic | Binary Search |
| O(n) | Linear | Duyệt mảng |
| O(n log n) | Linearithmic | Merge Sort, Quick Sort |
| O(n²) | Quadratic | Bubble Sort, Selection Sort |
| O(2ⁿ) | Exponential | Fibonacci đệ quy |
| O(n!) | Factorial | Permutations |

### Space Complexity (Độ Phức Tạp Không Gian)
Đo lường bộ nhớ mà thuật toán sử dụng.

```python
# O(1) - Không gian hằng số
def sum_array(arr):
    total = 0
    for num in arr:
        total += num
    return total

# O(n) - Không gian tuyến tính
def copy_array(arr):
    return arr.copy()
```

---

## 🗂️ Cấu Trúc Dữ Liệu Cơ Bản

### 1. Array (Mảng) & List

```python
# Khởi tạo
arr = [1, 2, 3, 4, 5]

# Các thao tác cơ bản
arr.append(6)        # O(1) - Thêm cuối
arr.insert(0, 0)     # O(n) - Thêm đầu
arr.pop()            # O(1) - Xóa cuối
arr.pop(0)           # O(n) - Xóa đầu
arr[2]               # O(1) - Truy cập
```

**Ứng dụng**: Lưu trữ dữ liệu tuần tự, cache, buffer

### 2. Linked List (Danh Sách Liên Kết)

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements
```

**Ưu điểm**: Thêm/xóa nhanh ở đầu (O(1))
**Nhược điểm**: Truy cập chậm (O(n))

### 3. Stack (Ngăn Xếp) - LIFO

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):      # Thêm vào đỉnh
        self.items.append(item)
    
    def pop(self):             # Lấy từ đỉnh
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):            # Xem đỉnh
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
```

**Ứng dụng**: 
- Undo/Redo trong text editor
- Back/Forward trong browser
- Function call stack
- Biểu thức toán học (infix to postfix)

### 4. Queue (Hàng Đợi) - FIFO

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):   # Thêm vào cuối
        self.items.append(item)
    
    def dequeue(self):         # Lấy từ đầu
        if not self.is_empty():
            return self.items.popleft()
        return None
    
    def is_empty(self):
        return len(self.items) == 0
```

**Ứng dụng**:
- BFS (Breadth-First Search)
- Task scheduling
- Print queue
- Message queue

### 5. Hash Table (Dictionary)

```python
# Python dict là hash table
phone_book = {
    "Alice": "0123456789",
    "Bob": "0987654321"
}

# Thao tác O(1) trung bình
phone_book["Charlie"] = "0111222333"  # Thêm
print(phone_book["Alice"])            # Truy cập
del phone_book["Bob"]                 # Xóa

# Xử lý collision với chaining
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def hash_function(self, key):
        return hash(key) % self.size
    
    def insert(self, key, value):
        hash_index = self.hash_function(key)
        for item in self.table[hash_index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[hash_index].append([key, value])
    
    def get(self, key):
        hash_index = self.hash_function(key)
        for item in self.table[hash_index]:
            if item[0] == key:
                return item[1]
        return None
```

**Ứng dụng**: Database indexing, cache, counting frequencies

---

## 🌳 Cấu Trúc Dữ Liệu Nâng Cao

### 1. Binary Tree (Cây Nhị Phân)

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    # Duyệt cây
    def inorder(self, node):      # Left -> Root -> Right
        if node:
            self.inorder(node.left)
            print(node.value, end=' ')
            self.inorder(node.right)
    
    def preorder(self, node):     # Root -> Left -> Right
        if node:
            print(node.value, end=' ')
            self.preorder(node.left)
            self.preorder(node.right)
    
    def postorder(self, node):    # Left -> Right -> Root
        if node:
            self.postorder(node.left)
            self.postorder(node.right)
            print(node.value, end=' ')
```

### 2. Binary Search Tree (BST)

```python
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None or node.value == value:
            return node
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)
```

**Độ phức tạp**: Insert/Search/Delete: O(log n) trung bình, O(n) worst case

### 3. Heap (Min/Max Heap)

```python
import heapq

# Min Heap (Python mặc định)
min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 7)
smallest = heapq.heappop(min_heap)  # 3

# Max Heap (dùng số âm)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
largest = -heapq.heappop(max_heap)  # 7
```

**Ứng dụng**: Priority Queue, Dijkstra's Algorithm, Heap Sort

### 4. Graph (Đồ Thị)

```python
# Adjacency List representation
class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    
    # BFS - Breadth First Search
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    # DFS - Depth First Search
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in self.graph.get(start, []):
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result
```

**Ứng dụng**: Social networks, maps, dependency graphs

### 5. Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**Ứng dụng**: Auto-complete, spell checker, IP routing

---

## 🔍 Thuật Toán Tìm Kiếm

### 1. Linear Search

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Time: O(n), Space: O(1)
```

### 2. Binary Search

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
# Yêu cầu: Mảng đã sắp xếp
```

---

## 📈 Thuật Toán Sắp Xếp

### 1. Bubble Sort

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Time: O(n²), Space: O(1)
```

### 2. Selection Sort

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Time: O(n²), Space: O(1)
```

### 3. Insertion Sort

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Time: O(n²), Space: O(1)
# Hiệu quả với mảng nhỏ hoặc gần sắp xếp
```

### 4. Merge Sort

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n), Space: O(n)
# Stable sort
```

### 5. Quick Sort

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Time: O(n log n) trung bình, O(n²) worst case
# Space: O(log n)
# Không stable
```

### 6. Heap Sort

```python
def heap_sort(arr):
    import heapq
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

# Time: O(n log n), Space: O(1)
```

### So Sánh Các Thuật Toán Sắp Xếp

| Thuật toán | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable |
|-----------|-------------|------------|--------------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |

---

## 🧩 Các Kỹ Thuật Quan Trọng

### 1. Two Pointers (Hai Con Trỏ)

```python
# Ví dụ: Tìm cặp số có tổng bằng target trong mảng đã sắp xếp
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None
```

### 2. Sliding Window (Cửa Sổ Trượt)

```python
# Ví dụ: Tìm subarray có tổng lớn nhất với độ dài k
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return None
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(len(arr) - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 3. Dynamic Programming (Quy Hoạch Động)

```python
# Ví dụ: Fibonacci với memoization
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# Ví dụ: Bài toán cái túi (Knapsack)
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]
```

### 4. Greedy Algorithm (Thuật Toán Tham Lam)

```python
# Ví dụ: Coin Change Problem
def min_coins(coins, amount):
    coins.sort(reverse=True)
    count = 0
    
    for coin in coins:
        if amount == 0:
            break
        count += amount // coin
        amount %= coin
    
    return count if amount == 0 else -1

# Ví dụ: Activity Selection
def activity_selection(start, finish):
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    selected = [activities[0]]
    
    for activity in activities[1:]:
        if activity[0] >= selected[-1][1]:
            selected.append(activity)
    
    return selected
```

### 5. Backtracking (Quay Lui)

```python
# Ví dụ: N-Queens Problem
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 1:
                return False
        
        # Check diagonals
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        for i, j in zip(range(row, -1, -1), range(col, n)):
            if board[i][j] == 1:
                return False
        
        return True
    
    def solve(board, row):
        if row >= n:
            return True
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                if solve(board, row + 1):
                    return True
                board[row][col] = 0
        
        return False
    
    board = [[0] * n for _ in range(n)]
    if solve(board, 0):
        return board
    return None

# Ví dụ: Sudoku Solver
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if solve():
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    solve()
    return board
```

### 6. Divide and Conquer (Chia Để Trị)

```python
# Ví dụ: Maximum Subarray (Kadane's Algorithm)
def max_subarray_sum(arr):
    max_current = max_global = arr[0]
    
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        max_global = max(max_global, max_current)
    
    return max_global

# Ví dụ: Merge Sort (đã trình bày ở trên)
```

---

## 🎓 Lộ Trình Học

### Giai Đoạn 1: Cơ Bản (2-3 tuần)
1. **Độ phức tạp thuật toán**: Big O, time/space complexity
2. **Array & List**: Operations, list comprehension
3. **String**: Manipulation, pattern matching
4. **Stack & Queue**: Implementation, applications
5. **Hash Table**: Dictionary, set operations

**Bài tập**: LeetCode Easy (20-30 bài)

### Giai Đoạn 2: Trung Cấp (4-6 tuần)
1. **Linked List**: Single, double, circular
2. **Recursion**: Base case, recursive thinking
3. **Sorting**: Bubble, selection, insertion, merge, quick
4. **Searching**: Linear, binary search
5. **Two Pointers & Sliding Window**
6. **Binary Tree**: Traversal, BST operations

**Bài tập**: LeetCode Medium (30-40 bài)

### Giai Đoạn 3: Nâng Cao (6-8 tuần)
1. **Dynamic Programming**: Memoization, tabulation
2. **Graph**: BFS, DFS, shortest path
3. **Heap**: Min/max heap, priority queue
4. **Greedy Algorithms**: Activity selection, Huffman coding
5. **Backtracking**: N-Queens, Sudoku, permutations
6. **Advanced Trees**: AVL, Red-Black, Trie
7. **Advanced Graph**: Dijkstra, Bellman-Ford, Floyd-Warshall

**Bài tập**: LeetCode Medium/Hard (40-50 bài)

### Giai Đoạn 4: Chuyên Sâu (Liên tục)
1. **Advanced DP**: Knapsack variations, LCS, LIS
2. **String Algorithms**: KMP, Rabin-Karp
3. **Advanced Graph**: Minimum Spanning Tree, Topological Sort
4. **Bit Manipulation**: Bitwise operations
5. **Math**: Number theory, combinatorics

**Bài tập**: Competitive programming, system design

---

## 📚 Tài Nguyên Học Tập

### Sách
- **"Grokking Algorithms"** - Aditya Bhargava (Dễ hiểu, có hình minh họa)
- **"Introduction to Algorithms"** - CLRS (Sách giáo khoa kinh điển)
- **"Cracking the Coding Interview"** - Gayle Laakmann McDowell
- **"Elements of Programming Interviews in Python"** - Adnan Aziz

### Website & Nền Tảng
- **LeetCode** (leetcode.com) - Kho bài tập khổng lồ, có discuss
- **HackerRank** (hackerrank.com) - Bài tập theo chủ đề
- **GeeksforGeeks** (geeksforgeeks.org) - Lý thuyết chi tiết
- **Visualgo** (visualgo.net) - Visualize algorithms
- **AlgoExpert** (algoexpert.io) - Video explanations (có phí)

### YouTube Channels
- **NeetCode** - Giải thích rõ ràng, có patterns
- **Back To Back SWE** - Chi tiết, chuyên sâu
- **Abdul Bari** - Lý thuyết algorithms
- **mycodeschool** - Data structures basics

### Courses
- **Coursera**: Algorithms Specialization (Stanford)
- **MIT OpenCourseWare**: Introduction to Algorithms
- **Udemy**: Python Data Structures & Algorithms
- **freeCodeCamp**: Data Structures (YouTube)

### Practice Platforms
- **LeetCode** - 2,500+ bài tập
- **CodeForces** - Competitive programming
- **AtCoder** - Competitive programming (Nhật)
- **Project Euler** - Math + algorithms

---

## 💡 Tips Học Hiệu Quả

### 1. Học Có Hệ Thống
- Đừng nhảy bài, học tuần tự từ dễ đến khó
- Mỗi ngày 1-2 bài, quan trọng là consistency
- Đọc kỹ lý thuyết trước khi làm bài tập

### 2. Practice Makes Perfect
- Code bằng tay trước khi chạy
- Giải lại các bài khó sau 1 tuần
- Đọc solutions của người khác để học cách tối ưu

### 3. Hiểu, Không Nhớ
- Tập trung hiểu logic, không học thuộc code
- Vẽ sơ đồ, visualize data flow
- Giải thích thuật toán bằng lời của bạn

### 4. Time Management
- Set timer 30-45 phút cho mỗi bài
- Nếu stuck, xem hint, rồi thử lại
- Sau 1 giờ vẫn không ra, xem solution và hiểu

### 5. Mock Interviews
- Practice coding trên whiteboard
- Nói to suy nghĩ của bạn (think aloud)
- Pramp, Interviewing.io cho mock interviews

### 6. Track Progress
- Ghi chép patterns đã học
- Tạo cheat sheet riêng
- Review weekly những gì đã học

---

## 🔥 Patterns Thường Gặp

### 1. Frequency Counter
Đếm tần suất xuất hiện, dùng dictionary
```python
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq
```

### 2. Multiple Pointers
Dùng nhiều con trỏ để duyệt mảng
```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### 3. Sliding Window
Cửa sổ di chuyển qua mảng
```python
def max_sum_subarray(arr, k):
    max_sum = sum(arr[:k])
    window_sum = max_sum
    for i in range(len(arr) - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### 4. Fast & Slow Pointers
Phát hiện cycle trong linked list
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### 5. In-place Reversal of Linked List
```python
def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

### 6. Tree BFS
Duyệt cây theo level
```python
from collections import deque

def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### 7. Tree DFS
Duyệt cây theo chiều sâu
```python
def dfs_inorder(root):
    if not root:
        return []
    return dfs_inorder(root.left) + [root.val] + dfs_inorder(root.right)
```

### 8. Top K Elements
Tìm K phần tử lớn/nhỏ nhất
```python
import heapq

def top_k_frequent(nums, k):
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    return heapq.nlargest(k, freq.keys(), key=freq.get)
```

### 9. Binary Search
Tìm kiếm trong sorted array
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 10. Subsets & Permutations
```python
def subsets(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result

def permutations(nums):
    if len(nums) <= 1:
        return [nums]
    result = []
    for i, num in enumerate(nums):
        remaining = nums[:i] + nums[i+1:]
        for perm in permutations(remaining):
            result.append([num] + perm)
    return result
```

---

## 🎯 Chuẩn Bị Phỏng Vấn

### Interview Process
1. **Behavioral Questions** (5-10 phút)
2. **Technical Questions** (30-45 phút)
   - 1-2 coding problems
   - Thảo luận approach
   - Code solution
   - Test cases
   - Optimize
3. **Q&A** (5-10 phút)

### Chiến Lược Giải Bài
1. **Clarify**: Đặt câu hỏi, hiểu rõ đề bài
2. **Example**: Vẽ ví dụ, identify pattern
3. **Approach**: Discuss brute force → optimal
4. **Code**: Viết code clean, có comments
5. **Test**: Run through test cases
6. **Optimize**: Thảo luận cải tiến

### Red Flags Cần Tránh
- ❌ Không đọc kỹ đề, làm ngay
- ❌ Im lặng, không communicate
- ❌ Code xong không test
- ❌ Bỏ qua edge cases
- ❌ Không discuss trade-offs

### Green Signals
- ✅ Ask clarifying questions
- ✅ Think aloud
- ✅ Start with simple approach
- ✅ Write clean, readable code
- ✅ Test with examples
- ✅ Discuss time/space complexity

---

## 🚀 Next Steps

1. **Bắt đầu với Foundations**: Học Big O và Array operations
2. **Code Daily**: 1-2 bài mỗi ngày, đừng bỏ lỡ
3. **Join Community**: LeetCode Discord, Reddit r/leetcode
4. **Build Projects**: Áp dụng DSA vào projects thực tế
5. **Mock Interviews**: Practice với bạn bè hoặc platforms

---

## 📊 Tracking Progress

### Checklist Cơ Bản
- [ ] Hiểu Big O notation
- [ ] Implement Array operations
- [ ] Implement Linked List
- [ ] Implement Stack & Queue
- [ ] Implement Hash Table
- [ ] Master Recursion
- [ ] Understand Binary Tree
- [ ] Solve 50 Easy problems
- [ ] Solve 30 Medium problems

### Checklist Nâng Cao
- [ ] Master Dynamic Programming
- [ ] Implement Graph algorithms
- [ ] Understand Greedy algorithms
- [ ] Master Backtracking
- [ ] Solve 20 Hard problems
- [ ] Complete mock interviews
- [ ] Build DSA projects

---

## 🌟 Kết Luận

Học DSA là một hành trình dài, nhưng mỗi bước đi đều đáng giá. Đừng nản lòng khi gặp khó khăn - mọi lập trình viên giỏi đều từng ở vị trí của bạn. 

**Key Takeaways:**
- Consistency là quan trọng nhất
- Hiểu logic, không học thuộc
- Practice, practice, practice!
- Learn from mistakes
- Enjoy the journey!

**Remember**: "The only way to learn a new programming language is by writing programs in it." - Dennis Ritchie

Good luck và chúc bạn thành công! 🎉

---

## 📞 Liên Hệ & Đóng Góp

Nếu bạn tìm thấy lỗi hoặc muốn đóng góp thêm nội dung, hãy tạo issue hoặc pull request.

**Happy Coding!** 💻✨
