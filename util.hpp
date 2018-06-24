#pragma once

#include <Eigen/Dense>

typedef float Scalar;

template<int A>
using vec = Eigen::Matrix<Scalar, A, 1>;

template<int A, int B>
using mat = Eigen::Matrix<Scalar, A, B>;
