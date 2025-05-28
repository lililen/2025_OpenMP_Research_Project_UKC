#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
//#define use_bilinear 1, for accidenal macros issues
constexpr bool use_bilinear = true;

#include <vector>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <immintrin.h>

namespace py = pybind11;

// helper function for reflection boundary conditions
inline int reflect_index(int idx, int size) {
    if (idx < 0) {
        return -idx - 1;
    } else if (idx >= size) {
        return 2 * size - idx - 1;
    }
    return idx;
}

// pixel access with reflection boundary conditions
inline float get_pixel_reflect(const float* ptr, int x, int y, int c, int w, int h, int channels) {
    int rx = reflect_index(x, w);
    int ry = reflect_index(y, h);
    return ptr[(ry * w + rx) * channels + c];
}

#if defined(__AVX2__)
inline __m256 bilinear_interp_avx2(
    __m256 v00, __m256 v01, __m256 v10, __m256 v11,
    __m256 fx, __m256 fy) {
    
    __m256 omfx = _mm256_sub_ps(_mm256_set1_ps(1.0f), fx);
    __m256 omfy = _mm256_sub_ps(_mm256_set1_ps(1.0f), fy);
    
    __m256 term1 = _mm256_mul_ps(_mm256_mul_ps(v00, omfx), omfy);
    __m256 term2 = _mm256_mul_ps(_mm256_mul_ps(v01, fx), omfy);
    __m256 term3 = _mm256_mul_ps(_mm256_mul_ps(v10, omfx), fy);
    __m256 term4 = _mm256_mul_ps(_mm256_mul_ps(v11, fx), fy);
    
    return _mm256_add_ps(_mm256_add_ps(term1, term2), 
                         _mm256_add_ps(term3, term4));
}
#endif

void rotate_omp(py::array_t<float> image, float deg) {
    auto buf = image.request();
    float* ptr = static_cast<float*>(buf.ptr);

    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    const int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    std::vector<float> output(w * h * channels, 0.0f);

    // float-> double for precision in angle calculations to match SciPy 
    double angle_rad = static_cast<double>(deg) * M_PI / 180.0;
    double sin_v = std::sin(angle_rad);
    double cos_v = std::cos(angle_rad);

    // SciPy uses (shape-1)/2.0 as center for proper pixel alignment
    double center_x = (w - 1) / 2.0;
    double center_y = (h - 1) / 2.0;

    //remove collapse(2) bc MSVC doesn't fully support it unless im using OpenMP 5+ with /openmp:11vm or /openmp:experimental
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) {
        //change to x+=8 for procrossing 8 pixels at the same time
        for (int x = 0; x < w; x++) {  // Process pixel by pixel for now to ensure correctness
            #if defined(__AVX2__)
            // disable AVX2 path for now to focus on correctness
            // will re-enable once we get SSIM=1.0 with scalar code
            if if (channels == 1 && x + 8 <= w) {
                // create vector of x coordinates [x, x+1, ..., x+7]
                __m256 x_vec = _mm256_add_ps(
                    _mm256_set1_ps(static_cast<float>(x)),
                    _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)
                );
                
                // convert to displacement from center
                __m256 dx_vec = _mm256_sub_ps(x_vec, _mm256_set1_ps(static_cast<float>(center_x)));
                __m256 dy_vec = _mm256_set1_ps(static_cast<float>(y - center_y));
                
                // rotation transformation
                __m256 src_x = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(cos_v)), dx_vec),
                        _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(-sin_v)), dy_vec)
                    ),
                    _mm256_set1_ps(static_cast<float>(center_x))
                );
                
                __m256 src_y = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(sin_v)), dx_vec),
                        _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(cos_v)), dy_vec)
                    ),
                    _mm256_set1_ps(static_cast<float>(center_y))
                );
                
                // Get integer and fractional parts
                __m256i x0 = _mm256_cvttps_epi32(src_x);
                __m256i y0 = _mm256_cvttps_epi32(src_y);
                __m256 fx = _mm256_sub_ps(src_x, _mm256_cvtepi32_ps(x0));
                __m256 fy = _mm256_sub_ps(src_y, _mm256_cvtepi32_ps(y0));
                
                // Gather pixels with reflection boundary handling
                alignas(32) int x0_arr[8], y0_arr[8];
                _mm256_store_si256((__m256i*)x0_arr, x0);
                _mm256_store_si256((__m256i*)y0_arr, y0);
                
                alignas(32) float v00_arr[8], v01_arr[8], v10_arr[8], v11_arr[8];
                
                for (int i = 0; i < 8; i++) {
                    int xi0 = reflect_index(x0_arr[i], w);
                    int yi0 = reflect_index(y0_arr[i], h);
                    int xi1 = reflect_index(x0_arr[i] + 1, w);
                    int yi1 = reflect_index(y0_arr[i] + 1, h);
                    
                    v00_arr[i] = ptr[yi0 * w + xi0];
                    v01_arr[i] = ptr[yi0 * w + xi1];
                    v10_arr[i] = ptr[yi1 * w + xi0];
                    v11_arr[i] = ptr[yi1 * w + xi1];
                }
                
                __m256 v00 = _mm256_load_ps(v00_arr);
                __m256 v01 = _mm256_load_ps(v01_arr);
                __m256 v10 = _mm256_load_ps(v10_arr);
                __m256 v11 = _mm256_load_ps(v11_arr);
                
                // Perform interpolation
                __m256 result = bilinear_interp_avx2(v00, v01, v10, v11, fx, fy);
                
                // Store results
                _mm256_storeu_ps(&output[y * w + x], result);
                
                x += 7; // Skip processed pixels
                continue;
            }
            #endif
   
            // Use double precision for coordinate transformations
            double dx = static_cast<double>(x) - center_x;
            double dy = static_cast<double>(y) - center_y;
            double src_x = cos_v * dx - sin_v * dy + center_x;
            double src_y = sin_v * dx + cos_v * dy + center_y;
            
            if (use_bilinear) {
                //bilinear interpolation
                // Ensure we stay within bounds for bilinear sampling
                // Get integer coordinates - use floor to match SciPy exactly
                int x0 = static_cast<int>(std::floor(src_x));
                int y0 = static_cast<int>(std::floor(src_y));
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // fractional parts - keep as double for precision
                double fx = src_x - static_cast<double>(x0);
                double fy = src_y - static_cast<double>(y0);

                // Bilinear interpolation with reflection boundary conditions
                //RGB has 3 channels-> change it to be arbitary number for channels
                for (int c = 0; c < channels; c++) {  // Fixed from c=3 to c<3
                    // pixel values as double for better precision calculation
                    //get_pixel_reflect matching with scipybenchmark
                    double v00 = static_cast<double>(get_pixel_reflect(ptr, x0, y0, c, w, h, channels));
                    double v01 = static_cast<double>(get_pixel_reflect(ptr, x1, y0, c, w, h, channels));
                    double v10 = static_cast<double>(get_pixel_reflect(ptr, x0, y1, c, w, h, channels));
                    double v11 = static_cast<double>(get_pixel_reflect(ptr, x1, y1, c, w, h, channels));

                    // higher precision bilinear interpolation calculation
                    double result = v00 * (1.0 - fx) * (1.0 - fy) +
                                   v01 * fx * (1.0 - fy) +
                                   v10 * (1.0 - fx) * fy +
                                   v11 * fx * fy;

                    output[(y * w + x) * channels + c] = static_cast<float>(result);
                }
            } else {
                // use nearest neighbor for edge cases to match SciPy boundary handling
                // nearest neighbor with reflection boundary conditions
                int nx = static_cast<int>(std::round(src_x));
                int ny = static_cast<int>(std::round(src_y));

                for (int c = 0; c < channels; c++) {
                    output[(y * w + x) * channels + c] = get_pixel_reflect(ptr, nx, ny, c, w, h, channels);
                }
            }
        }
    }
   
    //memcpy(ptr, output.data(), sizeof(float) * w * h * 3);
    memcpy(ptr, output.data(), sizeof(float) * w * h * channels);
}

PYBIND11_MODULE(rotate, m) {
    m.doc() = "image rotation with reflection boundary conditions";
    m.def("rotate", &rotate_omp,
        py::arg("img"), py::arg("degree"), "rotate img in parallel via OpenMP with bilinear interpolation and reflection boundaries");
}