#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
//#define use_bilinear 1, for accidenal macros issues
constexpr bool use_bilinear = true;

#include <vector>
#include <cmath>
#include <math.h>
#include <mutex>
#include <limits>
#include <algorithm>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

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
        for (int x = 0; x < w; x++) {
            // Use double precision for coordinate transformations
            double dx = static_cast<double>(x) - center_x;
            double dy = static_cast<double>(y) - center_y;
            double src_x = cos_v * dx - sin_v * dy + center_x;
            double src_y = sin_v * dx + cos_v * dy + center_y;

            if (use_bilinear) {
                //bilinear interpolation
                // Ensure we stay within bounds for bilinear sampling
                // Get integer coordinates
                int x0 = static_cast<int>(std::floor(src_x));
                int y0 = static_cast<int>(std::floor(src_y));
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // fractional parts
                double fx = src_x - x0;
                double fy = src_y - y0;

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