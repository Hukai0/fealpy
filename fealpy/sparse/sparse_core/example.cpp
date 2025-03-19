#include <pybind11/pybind11.h>

// 示例函数：计算两个数的和
int add(int a, int b) {
    return a + b;
}

// 示例类：矩形面积计算
class Rectangle {
public:
    Rectangle(double width, double height) : width_(width), height_(height) {}
    
    double area() const {
        return width_ * height_;
    }

private:
    double width_;
    double height_;
};

// Pybind11绑定代码
namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    // 绑定函数
    m.def("add", &add, "Add two integers");

    // 绑定类
    py::class_<Rectangle>(m, "Rectangle")
        .def(py::init<double, double>())  // 构造函数
        .def("area", &Rectangle::area);    // 成员函数
}

