#include <iostream>
#include <fstream>

/* fmt should be always included before Eigen headers because these latters
   may include <complex.h> from lapacke and it defines I that conflict with
   the fmt library. An alternative solution is to use LAPACK_COMPLEX_CUSTOM
   to prevent including <complex.h> */
#include <fmt/core.h>

template <typename VectorType, typename Function>
void WriteVectorAsRow(std::ofstream& f, const VectorType& v, Function fn) {
    const int n = v.rows();
    for (int i = 0; i < n; i++) {
        const double value = fn(v(i));
        f << fmt::format(" {:g}", value);
    }
    f << std::endl;
}

template <typename ArrayType>
void WriteMatrixData(const std::string& filename, const ArrayType& m) {
    std::ofstream f(filename, std::ios::binary);
    const int n1 = m.rows(), n2 = m.cols();
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            f << fmt::format(" {:g}", m(i, j));
        }
        f << std::endl;
    }
}

template <typename MatrixType>
void WriteComplexMatrixData(const std::string& filename, const MatrixType& m) {
    std::ofstream f(filename, std::ios::binary);
    const int n1 = m.rows(), n2 = m.cols();
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            const auto z = m(i, j);
            f << fmt::format(" {:g} {:g}", std::real(z), std::imag(z));
        }
        f << std::endl;
    }
}

