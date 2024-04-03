import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time

# Number of clusters (excluding paper strips)
n_colors = 2

# Load the image
china = cv2.imread("input/test.jpg", cv2.IMREAD_COLOR)
china = cv2.cvtColor(china, cv2.COLOR_BGR2RGB)

# Convert image to floats instead of the default 8-bit integer coding
china = np.array(china, dtype=np.float64) / 255

# Reshape the image into a 2D numpy array
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

# Perform k-means clustering
print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = image_array[np.random.choice(image_array.shape[0], 1000, replace=False)]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("Done fitting in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("Done predicting in %0.3fs." % (time() - t0))

# Identify the label corresponding to the yellow background
# We assume the background is the largest cluster
background_label = np.argmax(np.bincount(labels))

# Create a mask to remove pixels corresponding to the background
mask = (labels == background_label).reshape((w, h))

# Apply the mask to the original image to remove the background
quantized_image = china.copy()
quantized_image[mask] = [0, 0, 0]  # Set background pixels to black

# Convert the image back to uint8 format
quantized_image_uint8 = (quantized_image * 255).astype(np.uint8)

# Save the image without the background as a JPEG
output_file = "output/quantized_image.jpg"
cv2.imwrite(output_file, cv2.cvtColor(quantized_image_uint8, cv2.COLOR_RGB2BGR))
print(f"Image without background saved as {output_file}")

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(china)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title('Image with Background Removed')

plt.show()