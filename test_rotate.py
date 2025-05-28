import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("C:/Users/lenmo/research_project")
import rotate

# test image
w, h = 1024, 1024
img = np.zeros((h, w, 3), dtype=np.float32)
img[20:44, 20:44, :] = 1.0  

rotated_img = img.copy()

start = time.perf_counter()
rotate.rotate(rotated_img, 30.0)
elapsed = time.perf_counter() - start

# SSIM calculation 
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
# grayscale image to trigger AVX2 path
gray_img = np.zeros((h, w), dtype=np.float32)
gray_img[300:700, 300:700] = 1.0
rotated_gray = gray_img.copy()

start = time.perf_counter()
rotate.rotate(rotated_gray, 30.0)
elapsed_avx = time.perf_counter() - start

try:
    ssim_gray = structural_similarity(gray_img, rotated_gray, win_size=3)
    print(f"SSIM (Grayscale): {ssim_gray:.4f}")
except ValueError as e:
    print(f"SSIM calculation skipped (Grayscale): {str(e)}")

print("Rotation complete (Grayscale / AVX2 path).")
print(f"Max value: {rotated_gray.max():.4f}")
print(f"Mean absolute difference: {np.abs(rotated_gray - gray_img).mean():.6f}")
print(f"Time (AVX2 path): {elapsed_avx:.4f}s")

# calculation
if elapsed_avx > 0:
    speedup = elapsed_scalar / elapsed_avx
    print(f"Speedup (scalar / AVX2): {speedup:.2f}x")


plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1).imshow(img)
plt.title("Original RGB")
plt.subplot(2, 2, 2).imshow(rotated_img)
plt.title("Rotated RGB")
plt.subplot(2, 2, 3).imshow(gray_img, cmap='gray')
plt.title("Original Grayscale")
plt.subplot(2, 2, 4).imshow(rotated_gray, cmap='gray')
plt.title("Rotated Grayscale")
plt.tight_layout()
plt.show()