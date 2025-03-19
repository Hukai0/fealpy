#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "csr_utils.h"

namespace py = pybind11;

// 这里是一个假设的 vector_to_numpy 函数定义，确保能正确转换
template <typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    return py::array_t<T>(vec.size(), vec.data());
}

// 模块绑定
void bind_csr_utils(py::module_ &m) {
    m.def("csr_row_index", [](py::array_t<int> rows, py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax) {
        // 传递给 csr_row_index 的数组大小转换
        auto rows_buf = rows.request();
        int *rows_ptr = static_cast<int *>(rows_buf.ptr);

        auto Ap_buf = Ap.request();
        int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);

        auto Aj_buf = Aj.request();
        int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);

        auto Ax_buf = Ax.request();
        double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);

        // 使用 size_t 类型
        std::vector<int> Bj;
        std::vector<double> Bx;

        // 显式转换为适当类型
        csr_row_index<int, double>(static_cast<int>(rows.size()), rows_ptr, Ap_ptr, Aj_ptr, Ax_ptr, Bj, Bx);

        return std::make_tuple(vector_to_numpy(Bj), vector_to_numpy(Bx));
    });

    m.def("csr_row_slice", [](int start, int stop, int step, py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax) {
        auto Ap_buf = Ap.request();
        int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);

        auto Aj_buf = Aj.request();
        int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);

        auto Ax_buf = Ax.request();
        double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);

        std::vector<int> Bj;
        std::vector<double> Bx;

        csr_row_slice(start, stop, step, Ap_ptr, Aj_ptr, Ax_ptr, Bj, Bx);

        return std::make_tuple(vector_to_numpy(Bj), vector_to_numpy(Bx));
    });
    
    m.def("csr_column_index1", [](py::array_t<int> col_idxs, int n_row, int n_col, py::array_t<int> Ap, py::array_t<int> Aj) {
        auto col_idxs_buf = col_idxs.request();
        int *col_idxs_ptr = static_cast<int *>(col_idxs_buf.ptr);

        auto Ap_buf = Ap.request();
        int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);

        auto Aj_buf = Aj.request();
        int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);

        std::vector<int> col_offsets;
        std::vector<int> Bp;

        // 显式调用模板函数
        csr_column_index1<int>(col_idxs.size(), col_idxs_ptr, n_row, n_col, Ap_ptr, Aj_ptr, col_offsets, Bp);

        return std::make_tuple(vector_to_numpy(col_offsets), vector_to_numpy(Bp));
    });

    m.def("csr_column_index2", [](py::array_t<int> col_order, py::array_t<int> col_offset, int nnz, py::array_t<int> Aj, py::array_t<double> Ax) {
        auto col_order_buf = col_order.request();
        int *col_order_ptr = static_cast<int *>(col_order_buf.ptr);

        auto col_offset_buf = col_offset.request();
        int *col_offset_ptr = static_cast<int *>(col_offset_buf.ptr);

        auto Aj_buf = Aj.request();
        int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);

        auto Ax_buf = Ax.request();
        double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);

        std::vector<int> Bj;
        std::vector<double> Bx;

        // 显式调用模板函数
        csr_column_index2<int, double>(col_order_ptr, col_offset_ptr, nnz, Aj_ptr, Ax_ptr, Bj, Bx);

        return std::make_tuple(vector_to_numpy(Bj), vector_to_numpy(Bx));
    });

    m.def("csr_extract_submatrix", [](py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax,
                                      py::array_t<int> rows, py::array_t<int> cols) {
        auto Ap_buf = Ap.request();
        int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);

        auto Aj_buf = Aj.request();
        int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);

        auto Ax_buf = Ax.request();
        double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);

        auto rows_buf = rows.request();
        int *rows_ptr = static_cast<int *>(rows_buf.ptr);

        auto cols_buf = cols.request();
        int *cols_ptr = static_cast<int *>(cols_buf.ptr);

        std::vector<int> Bp, Bj;
        std::vector<double> Bx;

        // 调用 C++ 版本的 csr_extract_submatrix
        std::vector<int> rows_vec(rows_ptr, rows_ptr + rows_buf.size);
        std::vector<int> cols_vec(cols_ptr, cols_ptr + cols_buf.size);

        csr_extract_submatrix<int, double>(rows_vec.size(), Ap_ptr, Aj_ptr, Ax_ptr, rows_vec, cols_vec, Bp, Bj, Bx);


        return std::make_tuple(vector_to_numpy(Bp), vector_to_numpy(Bj), vector_to_numpy(Bx));
    });

    m.def("csr_find_nonzero", [](py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax) {
        // 获取 CSR 矩阵的行数
        auto Ap_buf = Ap.request();
        int* Ap_ptr = static_cast<int*>(Ap_buf.ptr);
        int n_row = Ap_buf.size - 1; // Ap 的大小为 n_row + 1

        auto Aj_buf = Aj.request();
        int* Aj_ptr = static_cast<int*>(Aj_buf.ptr);

        auto Ax_buf = Ax.request();
        double* Ax_ptr = static_cast<double*>(Ax_buf.ptr);

        // 结果容器
        std::vector<int> row_indices;
        std::vector<int> col_indices;
        std::vector<double> values;

        // 调用 C++ 实现
        csr_find_nonzero<int, double>(n_row, Ap_ptr, Aj_ptr, Ax_ptr, row_indices, col_indices, values);

        // 返回 numpy 数组
        return std::make_tuple(vector_to_numpy(row_indices), vector_to_numpy(col_indices), vector_to_numpy(values));
    });

        // CSR 矩阵转置
        m.def("csr_transpose", [](py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax, int n_row, int n_col) {
            // 读取输入数组数据
            auto Ap_buf = Ap.request();
            auto Aj_buf = Aj.request();
            auto Ax_buf = Ax.request();
    
            const int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);
            const int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);
            const double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);
    
            // 输出数组
            std::vector<int> Bp, Bj;
            std::vector<double> Bx;
    
            // 计算转置矩阵
            csr_transpose<int, double>(n_row, n_col, Ap_ptr, Aj_ptr, Ax_ptr, Bp, Bj, Bx);
    
            // 返回 numpy 数组
            return std::make_tuple(vector_to_numpy(Bp), vector_to_numpy(Bj), vector_to_numpy(Bx));
        });
    
        // CSR 矩阵每列最小值
        m.def("csr_column_min", [](py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax, int n_row, int n_col) {
            // 读取输入数组数据
            auto Ap_buf = Ap.request();
            auto Aj_buf = Aj.request();
            auto Ax_buf = Ax.request();
    
            const int *Ap_ptr = static_cast<int *>(Ap_buf.ptr);
            const int *Aj_ptr = static_cast<int *>(Aj_buf.ptr);
            const double *Ax_ptr = static_cast<double *>(Ax_buf.ptr);
    
            // 计算最小值
            std::vector<double> min_values;
            csr_column_min<int, double>(n_row, n_col, Ap_ptr, Aj_ptr, Ax_ptr, min_values);
    
            // 返回 numpy 数组
            return vector_to_numpy(min_values);
        });

}

PYBIND11_MODULE(csr_utils, m) {
    bind_csr_utils(m);
}
