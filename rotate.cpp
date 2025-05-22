#define _USE_MATH_DEFINES
#define _POSIX_C_SOURCE 200809L
#include <vector>
#include<cmath>
#include <math.h>
#include <mutex>
#include <cstring>
#include <pybind11/pybind11.h>
#include <omp.h>
namespace py =pybind11;
void rotate_omp(float *image, int w, int h, float deg){
    float angle = deg * M_PI/180.0f;
    float sin_v = sin(angle);
    float cos_v = cos(angle);
    float center_x = w/2.0f;
    float center_y = h/2.0f;

    #pragma omp parallel for collaspe(2) schedule(dynamic, 4)
    for (int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            float dx = x-center_x;
            float dy = y-center_y;
            int src_x = round(cos_v*dx -sin_v*dy+center_x);
            int src_y = round(sin_v*dx -cos_v*dy+center_y);


            if (src_x >= 0 && src_x <w && src_y >= 0 && src_y< h){
                #pragma omp simd
                for (int c=0; c<3; c++){
                    image[(y*w+x)*3 +c] = image[(src_y*w + src_x)*3+c];

                }
            }
        }
    }
}

PYBIND11_MODULE(test, m) {
    m.def("rotating", &rotate_omp, "parallel image rotation");
    }
