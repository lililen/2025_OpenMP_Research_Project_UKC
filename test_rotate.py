import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("C:/Users/lenmo/research_project")
import rotate

#white square 64x64 on black 
w, h = 64, 64
img = np.zeros((h, w, 3), dtype=np.float32)
img[20:44, 20:44, :] = 1.0  

rotated_img = img.copy()

# rotate+timer
start = time.perf_counter()
rotate.rotate(rotated_img, w, h, 30.0)  
elapsed = time.perf_counter() - start

print("Rotation complete.")
print(f"Max value after rotation: {rotated_img.max():.4f}")
print(f"Mean absolute difference from original: {np.abs(rotated_img - img).mean():.6f}")
print(f"Time taken: {elapsed:.4f}s")

# matplotlib visual
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Rotated (30°)")
plt.imshow(rotated_img)
plt.axis('off')

plt.tight_layout()
plt.show()
