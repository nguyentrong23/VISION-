import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Tập dữ liệu
data = np.array([[205, 181], [206, 181], ...])  # Điền tiếp dữ liệu vào đây

# Xác định số lượng đường thẳng bạn muốn phân loại
num_lines = ...

# Áp dụng thuật toán K-means để phân cụm
kmeans = KMeans(n_clusters=num_lines)
kmeans.fit(data)

# Nhãn của từng điểm tương ứng với cụm
labels = kmeans.labels_

# Tách riêng từng đường thẳng
lines = {}
for i in range(num_lines):
    lines[i] = data[labels == i]

# Hiển thị kết quả
for i in range(num_lines):
    plt.scatter(lines[i][:, 0], lines[i][:, 1], label=f'Line {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', label='Centroids')
plt.legend()
plt.show()

