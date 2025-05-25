import time
import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from rotate import rotate  # Matches your PYBIND11_MODULE name
from skimage.metrics import structural_similarity as ssim
 
def benchmark():
    sizes = [(256,256), (512,512), (1024,768), (768,1024)]
    results = []

    print("checking if benchmark is loading")

    for h, w in sizes:
        print(f"benchmark for size {h}x{w}")
        img = np.random.rand(h, w, 3).astype(np.float32)
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")

        try:
            # run OpenMP rotation 
            img_omp = img.copy()
            start = time.perf_counter()
            rotate(img_omp, 30.0)
            omp_time = time.perf_counter() - start

            # run SciPy rotation
            img_scipy = img.copy()
            start = time.perf_counter()
            scipy_rotated = scipy_rotate(img_scipy, 30, axes=(1,0), reshape=False, order=1)
            scipy_time = time.perf_counter() - start

            # diff check
            diff = np.mean(np.abs(img_omp - scipy_rotated))

            results.append({
                'size': f"{h}x{w}",
                'omp_time': omp_time,
                'scipy_time': scipy_time,
                'speedup': scipy_time / omp_time if omp_time > 0 else float('inf'),
                'diff': diff
            })
        #checking for errors
            print(f"appended results for {h}x{w}")

        except Exception as e:
            print(f"error on size {h}x{w}: {e}")
            break

    
    print(f"\n{'Size':<10} {'OpenMP (s)':<12} {'SciPy (s)':<12} {'Speedup':<12} {'Diff':<12}")
    for r in results:
        print(f"{r['size']:<10} {r['omp_time']:.6f}  {r['scipy_time']:.6f}  {r['speedup']:6.1f}x      {r['diff']:.6f}")

if __name__ == "__main__":
    benchmark()
