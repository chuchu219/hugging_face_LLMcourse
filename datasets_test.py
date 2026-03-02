import tensorflow as tf

# 載入資料
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 查看形狀與資料型態

print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

print("\nData type of train_images:", train_images.dtype)

# 顯示範例圖片與標籤
import matplotlib.pyplot as plt

plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()

# 查看像素值的範圍
print("Min pixel value:", train_images.min())
print("Max pixel value:", train_images.max())
