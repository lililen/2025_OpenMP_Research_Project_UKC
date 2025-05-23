#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
#include <vector>
#include<cmath>
#include <math.h>
#include <mutex>
#include <iostream>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py =pybind11;


void rotate_omp(py::array_t<float> image, int w, int h, float deg){
    auto buf = image.request();
    float*ptr = static_cast<float*>(buf.ptr);

    std::vector<float>output(w*h*3, 0.0f);
    float angle = (deg * M_PI/180.0);
    float sin_v = sin(angle);
    float cos_v = cos(angle);
    float center_x = w/2.0f;
    float center_y = h/2.0f;
//remove collapse(2) bc MSVC doesn't fully support it unless im using OpenMP 5+ with /openmp:11vm or /openmp:experimental
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            float dx = x-center_x;
            float dy = y-center_y;
            int src_x = static_cast<int>(round(cos_v*dx -sin_v*dy+center_x));
            int src_y = static_cast<int>(round(sin_v*dx +cos_v*dy+center_y));


            if (src_x >= 0 && src_x <w && src_y >= 0 && src_y< h){
                //optional remove bc error sys it canot vectorize it. remove unless optimizing for simd gains and benchmarking
                //#pragma omp simd
                for (int c=0; c<3; c++){
                    //in place modfication-> bad bc it messes up the pixels
                    output[(y*w+x)*3 +c] = ptr[(src_y*w + src_x)*3+c];

                }
            }
        }
    }
        memcpy(ptr, output.data(), sizeof(float) * w * h * 3);

}

PYBIND11_MODULE(rotate, m) {
    m.doc()="image rotation";
    m.def("rotate", &rotate_omp,
    py::arg("img"), py::arg("width"),py::arg("height"), py::arg("degree"), "rotate img in parallel vis OpenMP");
}
