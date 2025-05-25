#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
//#define use_bilinear 1, for accidenal macros issues
constexpr bool use_bilinear = true;


#include <vector>
#include<cmath>
#include <math.h>
#include <mutex>
#include <limits>
#include <algorithm>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py =pybind11;


void rotate_omp(py::array_t<float> image, float deg){
    auto buf = image.request();
    float*ptr = static_cast<float*>(buf.ptr);
    py::ssize_t h_raw = buf.shape[0];
    if (h_raw > std::numeric_limits<int>::max()) throw std::overflow_error("Image height too large");
    const int h = static_cast<int>(h_raw);

    py::ssize_t w_raw = buf.shape[1];
    if (w_raw > std::numeric_limits<int>::max()) throw std::overflow_error("Image width too large");
    const int w = static_cast<int>(w_raw);
    
    const int channels = buf.ndim == 3 ? buf.shape[2]: 1;

    std::vector<float>output(w*h*channels, 0.0f);
    float angle = deg * static_cast<float>(M_PI / 180.0);     float sin_v = sin(angle);
    float cos_v = cos(angle);
    float center_x = w/2.0f;
    float center_y = h/2.0f;
//remove collapse(2) bc MSVC doesn't fully support it unless im using OpenMP 5+ with /openmp:11vm or /openmp:experimental
    #pragma omp parallel for schedule(static) 
    for (int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            float dx = x-center_x;
            float dy = y-center_y;
            float src_x = cos_v * dx + sin_v * dy + center_x;
            float src_y = -sin_v * dx + cos_v * dy + center_y;

            
           //if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
            if (use_bilinear && src_x >= 1 && src_x < w-2 && src_y >= 1 && src_y < h-2) {                // Manual unrolling for RGB channels
                //output[(y * w + x) * 3 + 0] = ptr[(src_y * w + src_x) * 3 + 0];
                //output[(y * w + x) * 3 + 1] = ptr[(src_y * w + src_x) * 3 + 1];
                //output[(y * w + x) * 3 + 2] = ptr[(src_y * w + src_x) * 3 + 2];
                //bilinear interpolation
                const float fx = src_x - floor(src_x);
                const float fy = src_y - floor(src_y);
                const int x0 = static_cast<int>(src_x);
                const int y0 = static_cast<int>(src_y);
                const int x1 = std::min(x0 + 1, w - 1);
                const int y1 = std::min(y0 + 1, h - 1);
             
                //RGB has 3 channels
                for (int c = 0; c < 3; c++) {  // Fixed from c=3 to c<3
                        float v00 = ptr[(y0 * w + x0) * 3 + c];
                        float v01 = ptr[(y0 * w + x1) * 3 + c];
                        float v10 = ptr[(y1 * w + x0) * 3 + c];
                        float v11 = ptr[(y1 * w + x1) * 3 + c];
                    
                    output[(y * w + x) * 3 + c] = 
                        v00 * (1-fx)*(1-fy) + 
                        v01 * fx*(1-fy) + 
                        v10 * (1-fx)*fy + 
                        v11 * fx*fy;
            }
        }
        //edge case
     else if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
                int nx = std::clamp(static_cast<int>(round(src_x)), 0, w - 1);
                int ny = std::clamp(static_cast<int>(round(src_y)), 0, h - 1);

                for (int c = 0; c < 3; ++c) {
                    output[(y * w + x) * 3 + c] = ptr[(ny * w + nx) * 3 + c];
                }
            }
        }
    }
        //memcpy(ptr, output.data(), sizeof(float) * w * h * 3);

        memcpy(ptr, &output[0], sizeof(float) * w * h * 3);

}

PYBIND11_MODULE(rotate, m) {
    m.doc()="image rotation";
    m.def("rotate", &rotate_omp,
    py::arg("img"), py::arg("degree"), "rotate img in parallel vis OpenMP");
}
