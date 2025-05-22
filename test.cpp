#include <pybind11/pybind11.h>  

PYBID11_MODULE(test, m) {m.def("hello", []() {return "ready";});}
