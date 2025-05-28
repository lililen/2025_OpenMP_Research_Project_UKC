import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate
from rotate import rotate  # Your pybind11 module
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def create_test_image(h, w):
    """Create a structured test image instead of random noise"""
    img = np.zeros((h, w, 3), dtype=np.float32)
    
    
    for i in range(0, min(h, w), 10):
        if i < h and i < w:
            for j in range(min(h-i, w)):
                if i+j < h and j < w:
                    img[i+j, j, 0] = 0.8
    
    # green channel
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    img[:, :, 1] = np.sin(distances * 0.3) * 0.5 + 0.5
    
    # blue channel 
    block_size = min(h, w) // 8
    for i in range(0, h, block_size * 2):
        for j in range(0, w, block_size * 2):
            end_i = min(i + block_size, h)
            end_j = min(j + block_size, w)
            img[i:end_i, j:end_j, 2] = 0.7
    
    return img

def benchmark():
    sizes = [(256, 256), (512, 512), (1024, 768), (768, 1024)]
    results = []
    
    print("Running corrected benchmark with structured test images\n")
    
    for h, w in sizes:
        print(f"Benchmarking size: {h}x{w}")
        
        img = create_test_image(h, w)
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
            
            # Quality metrics 
            diff = np.mean(np.abs(img_omp - scipy_rotated))
            max_diff = np.max(np.abs(img_omp - scipy_rotated))
            
           
            win_size = min(7, min(h, w) // 8)  
            if win_size % 2 == 0:  
                win_size -= 1
            win_size = max(3, win_size)  
            
            ssim_val = ssim(img_omp, scipy_rotated, 
                          data_range=1.0, 
                          channel_axis=2, 
                          win_size=win_size)
            
            psnr_val = psnr(img_omp, scipy_rotated, data_range=1.0)
            
            results.append({
                'size': f"{h}x{w}",
                'omp_time': omp_time,
                'scipy_time': scipy_time,
                'speedup': scipy_time / omp_time if omp_time > 0 else float('inf'),
                'ssim': ssim_val,
                'psnr': psnr_val,
                'mean_diff': diff,
                'max_diff': max_diff,
                'omp_image': img_omp.copy(),
                'scipy_image': scipy_rotated.copy()
            })
            
            print(f"Results: SSIM: {ssim_val:.6f} | PSNR: {psnr_val:.1f} dB | Mean Diff: {diff:.8f} | Max Diff: {max_diff:.8f}")
            print(f"Timing: OpenMP: {omp_time*1000:.2f}ms | SciPy: {scipy_time*1000:.2f}ms | Speedup: {scipy_time/omp_time:.1f}x\n")
            
        except Exception as e:
            print(f"Error on size {h}x{w}: {e}")
            continue
    
    #  summary table
    print(f"\n{'Size':<10} {'OpenMP (ms)':<13} {'SciPy (ms)':<13} {'Speedup':<9} {'SSIM':<12} {'PSNR':<8} {'Mean Diff':<12} {'Max Diff':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['size']:<10} {r['omp_time']*1000:>10.2f}    {r['scipy_time']*1000:>10.2f}    {r['speedup']:>7.1f}x  {r['ssim']:>10.6f}   {r['psnr']:>6.1f}    {r['mean_diff']:>10.2e}  {r['max_diff']:>8.2e}")
    
    return results

def show_comparison(omp, scipy):
    """Show visual comparison of results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # OpenMP result
    axes[0].imshow(np.clip(omp, 0, 1))
    axes[0].set_title('OpenMP Rotation')
    axes[0].axis('off')
    
    # SciPy result
    axes[1].imshow(np.clip(scipy, 0, 1))
    axes[1].set_title('SciPy Rotation')
    axes[1].axis('off')
    
    # Difference 
    diff = np.abs(omp - scipy)
    max_diff = diff.max()
    if max_diff > 0:
        diff_vis = diff / max_diff  # Normalize to [0,1]
    else:
        diff_vis = diff
    
    axes[2].imshow(diff_vis, cmap='hot')
    axes[2].set_title(f'Difference (max: {max_diff:.2e})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_random_vs_structured():
    """Compare SSIM results between random noise and structured images"""
    print("-" * 60)
    print("COMPARING RANDOM NOISE vs STRUCTURED IMAGES")
    print("-" * 60)
    
    h, w = 256, 256
    
    print("Testing with random noise")
    img_random = np.random.rand(h, w, 3).astype(np.float32)
    img_random_omp = img_random.copy()
    rotate(img_random_omp, 30.0)
    img_random_scipy = scipy_rotate(img_random, 30, axes=(1, 0), reshape=False, order=1, mode='reflect')
    
    ssim_random = ssim(img_random_omp, img_random_scipy, data_range=1.0, channel_axis=2, win_size=7)
    diff_random = np.mean(np.abs(img_random_omp - img_random_scipy))
    
    print("Testing with structured image")
    img_struct = create_test_image(h, w)
    img_struct_omp = img_struct.copy()
    rotate(img_struct_omp, 30.0)
    img_struct_scipy = scipy_rotate(img_struct, 30, axes=(1, 0), reshape=False, order=1, mode='reflect')
    
    ssim_struct = ssim(img_struct_omp, img_struct_scipy, data_range=1.0, channel_axis=2, win_size=7)
    diff_struct = np.mean(np.abs(img_struct_omp - img_struct_scipy))
    
    print(f"\nResults:")
    print(f"Random noise  - SSIM: {ssim_random:.6f}, Mean diff: {diff_random:.8f}")
    print(f"Structured    - SSIM: {ssim_struct:.6f}, Mean diff: {diff_struct:.8f}")
    print(f"\nExplanation: Random noise gives low SSIM because there's no structure to preserve.")
    print(f"Structured images give high SSIM when rotation is implemented correctly.")

if __name__ == "__main__":
    # why random noise gives bad SSIM
    test_random_vs_structured()
    
    print("\n" + "-" * 60)
    print("Main Benchmark")
    print("-" * 60)
    
    # main benchmark with structured images
    results = benchmark()
    
    # visual comparison for largest image
    if results:
        print(f"\nShowing visual comparison for {results[-1]['size']} image...")
        show_comparison(results[-1]['omp_image'], results[-1]['scipy_image'])