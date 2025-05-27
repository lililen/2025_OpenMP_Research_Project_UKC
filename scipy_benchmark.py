import time
import numpy as np  # You had: `import numpy as` (incomplete)
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate
from rotate import rotate  # Your pybind11 module
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def benchmark():
    sizes = [(256,256), (512,512), (1024,768), (768,1024)]
    results = []

    print("Checking if benchmark is loading\n")

    for h, w in sizes:
        print(f"Benchmarking size: {h}x{w}")
        img = np.random.rand(h, w, 3).astype(np.float32)
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")

        try:
            # OpenMP rotation
            img_omp = img.copy()
            start = time.perf_counter()
            rotate(img_omp, 30.0)
            omp_time = time.perf_counter() - start

            # SciPy rotation
            img_scipy = img.copy()
            start = time.perf_counter()
            scipy_rotated = scipy_rotate(img_scipy, 30, axes=(1, 0), reshape=False, order=1, mode='reflect')
            scipy_time = time.perf_counter() - start

            # difference and quality metrics
            diff = np.mean(np.abs(img_omp - scipy_rotated))
            ssim_val = ssim(img_omp, scipy_rotated, data_range=1.0, channel_axis=2, win_size=3)
            psnr_val = psnr(img_omp, scipy_rotated, data_range=1.0)

            results.append({
                'size': f"{h}x{w}",
                'omp_time': omp_time,
                'scipy_time': scipy_time,
                'speedup': scipy_time / omp_time if omp_time > 0 else float('inf'),
                'ssim': ssim_val,
                'psnr': psnr_val,
                'diff': diff,
                'omp_image': img_omp.copy(),  # Store for visual checks
                'scipy_image': scipy_rotated.copy()
            })

            print(f"benchmark for size {h}x{w} | SSIM: {ssim_val:.3f} | PSNR: {psnr_val:.1f} dB | Diff: {diff:.4f}\n")

        except Exception as e:
            print(f"Error on size {h}x{w}: {e}")
            continue

    print(f"\n{'Size':<10} {'OpenMP (ms)':<13} {'SciPy (ms)':<13} {'Speedup':<9} {'SSIM':<7} {'PSNR':<6} {'Diff':<8}")
    for r in results:
        print(f"{r['size']:<10} {r['omp_time']*1000:>10.3f}    {r['scipy_time']*1000:>10.3f}    {r['speedup']:>7.1f}x  {r['ssim']:>6.3f}   {r['psnr']:>5.1f}  {r['diff']:>7.4f}")

    return results  # visualize later

def show_comparison(omp, scipy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(omp)
    ax1.set_title('OpenMP Rotation')
    ax1.axis('off')
    ax2.imshow(scipy)
    ax2.set_title('SciPy Rotation')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = benchmark()

    # visual comparison for largest image
    if results:
        show_comparison(results[-1]['omp_image'], results[-1]['scipy_image'])
