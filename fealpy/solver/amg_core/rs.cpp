#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>  // std::fill, std::swap

namespace py = pybind11;

constexpr int U_NODE = -1;
constexpr int F_NODE = 0;
constexpr int C_NODE = 1;
constexpr int PRE_F_NODE = 2;

template <typename I>
py::array_t<I> rs_cf_splitting(
    py::array_t<I> Sp, py::array_t<I> Sj,
    py::array_t<I> Tp, py::array_t<I> Tj) {

    // 获取各数组的缓冲区信息
    py::buffer_info Sp_buf = Sp.request();
    py::buffer_info Sj_buf = Sj.request();
    py::buffer_info Tp_buf = Tp.request();
    py::buffer_info Tj_buf = Tj.request();

    // 假设 n_nodes = Tp 长度 - 1（也可以用 Sp_buf.shape[0]-1）
    I n_nodes = static_cast<I>(Tp_buf.shape[0]) - 1;

    // 创建输出数组 splitting，大小为 n_nodes
    auto splitting = py::array_t<I>(n_nodes);
    py::buffer_info splitting_buf = splitting.request();

    // 获取各数组数据指针
    I* Sp_ptr = static_cast<I*>(Sp_buf.ptr);
    I* Sj_ptr = static_cast<I*>(Sj_buf.ptr);
    I* Tp_ptr = static_cast<I*>(Tp_buf.ptr);
    I* Tj_ptr = static_cast<I*>(Tj_buf.ptr);
    I* splitting_ptr = static_cast<I*>(splitting_buf.ptr);

    // 计算每个节点的 lambda 值（这里仅用 Tp 差值）
    std::vector<I> lambda(n_nodes, 0);
    I lambda_max = 0;
    for (I i = 0; i < n_nodes; i++) {
        lambda[i] = Tp_ptr[i + 1] - Tp_ptr[i];
        if (lambda[i] > lambda_max) {
            lambda_max = lambda[i];
        }
    }
    lambda_max *= 2;
    if (n_nodes + 1 > lambda_max) {
        lambda_max = n_nodes + 1;
    }

    // 初始化区间相关数组与节点映射
    std::vector<I> interval_ptr(lambda_max, 0);
    std::vector<I> interval_count(lambda_max, 0);
    std::vector<I> index_to_node(n_nodes);
    std::vector<I> node_to_index(n_nodes);

    // 统计每个 lambda 值对应的节点数量
    for (I i = 0; i < n_nodes; i++) {
        interval_count[lambda[i]]++;
    }
    for (I i = 0, cumsum = 0; i < lambda_max; i++) {
        interval_ptr[i] = cumsum;
        cumsum += interval_count[i];
        interval_count[i] = 0;
    }
    for (I i = 0; i < n_nodes; i++) {
        I lambda_i = lambda[i];
        I index = interval_ptr[lambda_i] + interval_count[lambda_i];
        index_to_node[index] = i;
        node_to_index[i] = index;
        interval_count[lambda_i]++;
    }

    // 将所有节点初始标记为 U_NODE
    std::fill(splitting_ptr, splitting_ptr + n_nodes, U_NODE);

    // 若节点无邻居或只有一个且自指，则标记为 F_NODE
    for (I i = 0; i < n_nodes; i++) {
        if (lambda[i] == 0 || (lambda[i] == 1 && Tj_ptr[Tp_ptr[i]] == i)) {
            splitting_ptr[i] = F_NODE;
        }
    }

    // 从高到低处理节点
    for (I top_index = n_nodes - 1; top_index >= 0; top_index--) {
        I i = index_to_node[top_index];
        I lambda_i = lambda[i];
        interval_count[lambda_i]--;  // 从当前区间移除节点

        if (lambda[i] <= 0)
            break;

        if (splitting_ptr[i] == U_NODE) {
            splitting_ptr[i] = C_NODE;

            // 遍历 Tp_ptr 范围内的所有邻接节点 j，将 U_NODE 标记为 PRE_F_NODE
            for (I jj = Tp_ptr[i]; jj < Tp_ptr[i + 1]; jj++) {
                I j = Tj_ptr[jj];
                if (splitting_ptr[j] == U_NODE) {
                    splitting_ptr[j] = PRE_F_NODE;
                }
            }

            // 对于刚标记为 PRE_F_NODE 的节点 j，改为 F_NODE，并调整其邻居 k 的 lambda 值
            for (I jj = Tp_ptr[i]; jj < Tp_ptr[i + 1]; jj++) {
                I j = Tj_ptr[jj];
                if (splitting_ptr[j] == PRE_F_NODE) {
                    splitting_ptr[j] = F_NODE;
                    // 遍历节点 j 的 S_j 邻接区间
                    for (I kk = Sp_ptr[j]; kk < Sp_ptr[j + 1]; kk++) {
                        I k = Sj_ptr[kk];
                        if (splitting_ptr[k] == U_NODE) {
                            // 若 lambda[k] 已达上界，则跳过
                            if (lambda[k] >= n_nodes - 1) {
                                continue;
                            }
                            I lambda_k = lambda[k];
                            I old_pos  = node_to_index[k];
                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;
                            // 将 k 移动到当前区间末尾：交换映射位置
                            std::swap(index_to_node[old_pos], index_to_node[new_pos]);
                            // 更新节点到索引的映射
                            node_to_index[index_to_node[old_pos]] = old_pos;
                            node_to_index[index_to_node[new_pos]] = new_pos;
                            // 更新区间计数和指针（注意边界检查）
                            interval_count[lambda_k] -= 1;
                            if (lambda_k + 1 < static_cast<I>(interval_count.size())) {
                                interval_count[lambda_k + 1] += 1;
                                interval_ptr[lambda_k + 1] = new_pos;
                            }
                            // 增加 lambda 值
                            lambda[k]++;
                        }
                    }
                }
            }
            // 对于节点 i 的邻接节点 j，减少 lambda[j] 值
            for (I jj = Sp_ptr[i]; jj < Sp_ptr[i+1]; jj++) {
                I j = Sj_ptr[jj];
                if (splitting_ptr[j] == U_NODE) {
                    if (lambda[j] == 0) {
                        continue;
                    }
                    I lambda_j = lambda[j];
                    I old_pos  = node_to_index[j];
                    I new_pos  = interval_ptr[lambda_j];
                    std::swap(index_to_node[old_pos], index_to_node[new_pos]);
                    node_to_index[index_to_node[old_pos]] = old_pos;
                    node_to_index[index_to_node[new_pos]] = new_pos;
                    interval_count[lambda_j] -= 1;
                    if (lambda_j - 1 < static_cast<I>(interval_count.size())) {
                        interval_count[lambda_j - 1] += 1;
                        interval_ptr[lambda_j] += 1;
                        interval_ptr[lambda_j - 1] = interval_ptr[lambda_j] - interval_count[lambda_j - 1];
                    }
                    lambda[j]--;
                }
            }
        }
    }

    // 将仍为 U_NODE 的节点标记为 F_NODE
    for (I i = 0; i < n_nodes; i++) {
        if (splitting_ptr[i] == U_NODE) {
            splitting_ptr[i] = F_NODE;
        }
    }

    return splitting;
}

template<class I>
void rs_direct_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Pp[], const int Pp_size)
{
    I nnz = 0;
    Pp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        } else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Pp[i+1] = nnz;
    }
}



template<class I, class T>
void rs_direct_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Pp[], const int Pp_size,
                                         I Pj[], const int Pj_size,
                                         T Px[], const int Px_size)
{

    for(I i = 0; i < n_nodes; i++){
        if(splitting[i] == C_NODE){
            Pj[Pp[i]] = i;
            Px[Pp[i]] = 1;
        } else {
            T sum_strong_pos = 0, sum_strong_neg = 0;
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    if (Sx[jj] < 0)
                        sum_strong_neg += Sx[jj];
                    else
                        sum_strong_pos += Sx[jj];
                }
            }

            T sum_all_pos = 0, sum_all_neg = 0;
            T diag = 0;
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                if (Aj[jj] == i){
                    diag += Ax[jj];
                } else {
                    if (Ax[jj] < 0)
                        sum_all_neg += Ax[jj];
                    else
                        sum_all_pos += Ax[jj];
                }
            }

            T alpha = sum_all_neg / sum_strong_neg;
            T beta  = sum_all_pos / sum_strong_pos;

            if (sum_strong_pos == 0){
                diag += sum_all_pos;
                beta = 0;
            }

            T neg_coeff = -alpha/diag;
            T pos_coeff = -beta/diag;

            I nnz = Pp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    Pj[nnz] = Sj[jj];
                    if (Sx[jj] < 0)
                        Px[nnz] = neg_coeff * Sx[jj];
                    else
                        Px[nnz] = pos_coeff * Sx[jj];
                    nnz++;
                }
            }
        }
    }


    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++){
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Pp[n_nodes]; i++){
        Pj[i] = map[Pj[i]];
    }
}

template<typename T>
int signof(T x, T eps = 1e-15) {
    if (x > eps) return 1;
    else if (x < -eps) return -1;
    else return 0;
}

template<class I, class T>
void rs_classical_interpolation_pass2(const I n_nodes,
                                      const I Ap[], const int Ap_size,
                                      const I Aj[], const int Aj_size,
                                      const T Ax[], const int Ax_size,
                                      const I Sp[], const int Sp_size,
                                      const I Sj[], const int Sj_size,
                                      const T Sx[], const int Sx_size,
                                      const I splitting[], const int splitting_size,
                                      const I Pp[], const int Pp_size,
                                            I Pj[], const int Pj_size,
                                            T Px[], const int Px_size,
                                      const bool modified)
{
    for (I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            Pj[Pp[i]] = i;
            Px[Pp[i]] = 1;
        }
        // Otherwise, use RS classical interpolation formula
        else {

            // Calculate denominator
            T denominator = 0;

            // Start by summing entire row of A
            for (I mm = Ap[i]; mm < Ap[i+1]; mm++) {
                denominator += Ax[mm];
            }

            // Then subtract off the strong connections so that you are left with
            // denominator = a_ii + sum_{m in weak connections} a_im
            for (I mm = Sp[i]; mm < Sp[i+1]; mm++) {
                if ( Sj[mm] != i ) {
                    denominator -= Sx[mm]; // making sure to leave the diagonal entry in there
                }
            }

            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = Pp[i];
            for (I jj = Sp[i]; jj < Sp[i+1]; jj++) {

                if (splitting[Sj[jj]] == C_NODE) {

                    // Set temporary value for Pj as global index, j. Will be mapped to
                    // appropriate coarse-grid column index after all data is filled in.
                    Pj[nnz] = Sj[jj];
                    I j = Sj[jj];

                    // Initialize numerator as a_ij
                    T numerator = Sx[jj];

                    // Sum over strongly connected fine points
                    for (I kk = Sp[i]; kk < Sp[i+1]; kk++) {
                        if ( (splitting[Sj[kk]] == F_NODE) && (Sj[kk] != i) ) {

                            // Get column k and value a_ik
                            I k = Sj[kk];
                            T a_ik = Sx[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            T a_kk = 0;
                            if (modified) {
                                for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                    if (Aj[search_ind] == j) {
                                        a_kj = Ax[search_ind];
                                    }
                                    else if (Aj[search_ind] == k) {
                                        a_kk = Ax[search_ind];
                                    }
                                }
                            }
                            else {
                                for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                    if ( Aj[search_ind] == j ) {
                                        a_kj = Ax[search_ind];
                                        break;
                                    }
                                }
                            }

                            // If sign of a_kj matches sign of a_kk, ignore a_kj in sum
                            // (i.e. leave as a_kj = 0) for modified interpolation
                            if ( modified && (signof(a_kj) == signof(a_kk)) ) {
                                a_kj = 0;
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (std::abs(a_kj) > 1e-15*std::abs(a_ik)) {

                                // Calculate sum for inner denominator (loop over strongly connected C-points)
                                T inner_denominator = 0;
                                for (I ll = Sp[i]; ll < Sp[i+1]; ll++) {
                                    if (splitting[Sj[ll]] == C_NODE) {

                                        // Get column l
                                        I l = Sj[ll];

                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        // Only add if sign of a_kl does not equal sign of a_kk
                                        for (I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                            if (Aj[search_ind] == l) {
                                                T a_kl = Ax[search_ind];
                                                if ( (!modified) || (signof(a_kl) != signof(a_kk)) ) {
                                                    inner_denominator += a_kl;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }

                                // Add a_ik * a_kj / inner_denominator to the numerator
                                if (std::abs(inner_denominator) < 1e-15*std::abs(a_ik * a_kj)) {
                                    printf("Inner denominator was zero.\n");
                                }
                                numerator += a_ik * a_kj / inner_denominator;
                            }
                        }
                    }

                    // Set w_ij = -numerator/denominator
                    if (std::abs(denominator) < 1e-15*std::abs(numerator)) {
                        printf("Outer denominator was zero: diagonal plus sum of weak connections was zero.\n");
                    }
                    Px[nnz] = -numerator / denominator;
                    nnz++;
                }
            }
        }
    }

    // Column indices were initially stored as global indices. Build map to switch
    // to C-point indices.
    std::vector<I> map(n_nodes);
    for (I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for (I i = 0; i < Pp[n_nodes]; i++) {
        Pj[i] = map[Pj[i]];
    }
}



PYBIND11_MODULE(amg_core, m) {
    m.def("rs_cf_splitting", &rs_cf_splitting<int>, "Ruge-Stueben CF Splitting (int)");
    m.def("rs_cf_splitting", &rs_cf_splitting<long>, "Ruge-Stueben CF Splitting (long)");
        // rs_direct_interpolation_pass1 绑定（示例绑定 int 类型）
        m.def("rs_direct_interpolation_pass1",
            [](int n_nodes,
               py::array_t<int> Sp, py::array_t<int> Sj,
               py::array_t<int> splitting, py::array_t<int> Pp) {
                auto Sp_buf = Sp.request();
                auto Sj_buf = Sj.request();
                auto splitting_buf = splitting.request();
                auto Pp_buf = Pp.request();
                int* Sp_ptr = static_cast<int*>(Sp_buf.ptr);
                int* Sj_ptr = static_cast<int*>(Sj_buf.ptr);
                int* splitting_ptr = static_cast<int*>(splitting_buf.ptr);
                int* Pp_ptr = static_cast<int*>(Pp_buf.ptr);
                rs_direct_interpolation_pass1(n_nodes,
                                              Sp_ptr, Sp_buf.size,
                                              Sj_ptr, Sj_buf.size,
                                              splitting_ptr, splitting_buf.size,
                                              Pp_ptr, Pp_buf.size);
            },
            "RS direct interpolation pass 1 (int)");
  
      // rs_direct_interpolation_pass2 绑定（示例绑定 I 为 int，T 为 double）
      m.def("rs_direct_interpolation_pass2",
            [](int n_nodes,
               py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax,
               py::array_t<int> Sp, py::array_t<int> Sj, py::array_t<double> Sx,
               py::array_t<int> splitting, py::array_t<int> Pp,
               py::array_t<int> Pj, py::array_t<double> Px) {
                auto Ap_buf = Ap.request();
                auto Aj_buf = Aj.request();
                auto Ax_buf = Ax.request();
                auto Sp_buf = Sp.request();
                auto Sj_buf = Sj.request();
                auto Sx_buf = Sx.request();
                auto splitting_buf = splitting.request();
                auto Pp_buf = Pp.request();
                auto Pj_buf = Pj.request();
                auto Px_buf = Px.request();
                int* Ap_ptr = static_cast<int*>(Ap_buf.ptr);
                int* Aj_ptr = static_cast<int*>(Aj_buf.ptr);
                double* Ax_ptr = static_cast<double*>(Ax_buf.ptr);
                int* Sp_ptr = static_cast<int*>(Sp_buf.ptr);
                int* Sj_ptr = static_cast<int*>(Sj_buf.ptr);
                double* Sx_ptr = static_cast<double*>(Sx_buf.ptr);
                int* splitting_ptr = static_cast<int*>(splitting_buf.ptr);
                int* Pp_ptr = static_cast<int*>(Pp_buf.ptr);
                int* Pj_ptr = static_cast<int*>(Pj_buf.ptr);
                double* Px_ptr = static_cast<double*>(Px_buf.ptr);
                rs_direct_interpolation_pass2(n_nodes,
                                              Ap_ptr, Ap_buf.size,
                                              Aj_ptr, Aj_buf.size,
                                              Ax_ptr, Ax_buf.size,
                                              Sp_ptr, Sp_buf.size,
                                              Sj_ptr, Sj_buf.size,
                                              Sx_ptr, Sx_buf.size,
                                              splitting_ptr, splitting_buf.size,
                                              Pp_ptr, Pp_buf.size,
                                              Pj_ptr, Pj_buf.size,
                                              Px_ptr, Px_buf.size);
            },
            "RS direct interpolation pass 2 (int, double)");

            m.def("rs_classical_interpolation_pass2",
                [](int n_nodes,
                   py::array_t<int> Ap, py::array_t<int> Aj, py::array_t<double> Ax,
                   py::array_t<int> Sp, py::array_t<int> Sj, py::array_t<double> Sx,
                   py::array_t<int> splitting, py::array_t<int> Pp,
                   py::array_t<int> Pj, py::array_t<double> Px,
                   bool modified) {
                      auto Ap_buf = Ap.request();
                      auto Aj_buf = Aj.request();
                      auto Ax_buf = Ax.request();
                      auto Sp_buf = Sp.request();
                      auto Sj_buf = Sj.request();
                      auto Sx_buf = Sx.request();
                      auto splitting_buf = splitting.request();
                      auto Pp_buf = Pp.request();
                      auto Pj_buf = Pj.request();
                      auto Px_buf = Px.request();
          
                      int* Ap_ptr = static_cast<int*>(Ap_buf.ptr);
                      int* Aj_ptr = static_cast<int*>(Aj_buf.ptr);
                      double* Ax_ptr = static_cast<double*>(Ax_buf.ptr);
                      int* Sp_ptr = static_cast<int*>(Sp_buf.ptr);
                      int* Sj_ptr = static_cast<int*>(Sj_buf.ptr);
                      double* Sx_ptr = static_cast<double*>(Sx_buf.ptr);
                      int* splitting_ptr = static_cast<int*>(splitting_buf.ptr);
                      int* Pp_ptr = static_cast<int*>(Pp_buf.ptr);
                      int* Pj_ptr = static_cast<int*>(Pj_buf.ptr);
                      double* Px_ptr = static_cast<double*>(Px_buf.ptr);
          
                      rs_classical_interpolation_pass2(n_nodes,
                                                       Ap_ptr, static_cast<int>(Ap_buf.size),
                                                       Aj_ptr, static_cast<int>(Aj_buf.size),
                                                       Ax_ptr, static_cast<int>(Ax_buf.size),
                                                       Sp_ptr, static_cast<int>(Sp_buf.size),
                                                       Sj_ptr, static_cast<int>(Sj_buf.size),
                                                       Sx_ptr, static_cast<int>(Sx_buf.size),
                                                       splitting_ptr, static_cast<int>(splitting_buf.size),
                                                       Pp_ptr, static_cast<int>(Pp_buf.size),
                                                       Pj_ptr, static_cast<int>(Pj_buf.size),
                                                       Px_ptr, static_cast<int>(Px_buf.size),
                                                       modified);
                },
                "RS classical interpolation pass 2 (int, double, bool)");
          
}
