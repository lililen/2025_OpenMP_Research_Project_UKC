import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("C:/Users/lenmo/research_project")
import rotate

# Create test image
w, h = 64, 64
img = np.zeros((h, w, 3), dtype=np.float32)
img[20:44, 20:44, :] = 1.0  # White square

rotated_img = img.copy()

# Rotate with timing
start = time.perf_counter()
rotate.rotate(rotated_img, 30.0)
elapsed = time.perf_counter() - start

# SSIM calculation (updated)
from skimage.metrics import structural_similarity
try:
    ssim = structural_similarity(img, rotated_img, channel_axis=-1, win_size=3)
    print(f"SSIM: {ssim:.4f}")
except ValueError as e:
    print(f"SSIM calculation skipped: {str(e)}")

print("Rotation complete.")
print(f"Max value: {rotated_img.max():.4f}")
print(f"Mean absolute difference: {np.abs(rotated_img - img).mean():.6f}")
print(f"Time: {elapsed:.4f}s")

# Visualization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1).imshow(img)
plt.subplot(1, 2, 2).imshow(rotated_img)
plt.show()