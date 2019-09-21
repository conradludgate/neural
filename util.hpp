#pragma once

#include <Eigen/Dense>

namespace neural
{

// template <int A>
// using vec = Eigen::Matrix<Scalar, A, 1>;

// template <int A, int B>
// using mat = Eigen::Matrix<Scalar, A, B>;

template <typename S, int A>
using vec = Eigen::Matrix<S, A, 1>;

template <typename S, int A, int B>
using mat = Eigen::Matrix<S, A, B>;

// template <typename Scalar, int Output, int Batch, typename Activation, typename Next>
// mat<Scalar, Output, Batch> next(mat<Scalar, Output, Batch> output, mat<Scalar, Next::Output, Batch> expected)
// {
//     return Next::template feedforwad_backward<Batch, decltype(Next::next())>(output, expected);
// }

// https://stackoverflow.com/a/39101723
template <typename Tuple, std::size_t... Is>
auto pop_front_impl(const Tuple &tuple, std::index_sequence<Is...>)
{
    return std::make_tuple(std::get<1 + Is>(tuple)...);
}

template <typename Tuple>
auto pop_front(const Tuple &tuple)
{
    return pop_front_impl(tuple,
                          std::make_index_sequence<std::tuple_size<Tuple>::value - 1>());
}

namespace activation
{

class Sigmoid
{
private:
    template <typename S>
    struct CwiseSigmoid
    {
        const S operator()(const S &x) const
        {
            return 1 / (exp(-x) + 1);
        }
    };

    template <typename S>
    struct SigmoidPrime
    {
        typedef S result_type;
        const S operator()(const S &a, const S &b) const
        {
            return a * (1 - b) * b;
        }
    };

public:
    template <typename S, int A, int B>
    const static mat<S, A, B> g(const mat<S, A, B> &v)
    {
        return v.unaryExpr(CwiseSigmoid<S>());
    }
    template <typename S, int A, int B>
    const static mat<S, A, B> gprime(const mat<S, A, B> &error, const mat<S, A, B> &output)
    {
        return error.binaryExpr(output, SigmoidPrime<S>());
    }
};

class Relu
{
private:
    template <typename S>
    struct CwiseRelu
    {
        const S operator()(const S &x) const
        {
            if (x > 0)
                return x;
            else
                return 0;
        }
    };

    template <typename S>
    struct ReluPrime
    {
        typedef S result_type;
        const S operator()(const S &a, const S &b) const
        {
            if (b > 0)
                return a;
            else
                return 0;
        }
    };

public:
    template <typename S, int A, int B>
    const static mat<S, A, B> g(const mat<S, A, B> &v)
    {
        return v.unaryExpr(CwiseRelu<S>());
    }
    template <typename S, int A, int B>
    const static mat<S, A, B> gprime(const mat<S, A, B> &error, const mat<S, A, B> &output)
    {
        return error.binaryExpr(output, ReluPrime<S>());
    }
};

} // namespace activation

namespace cost
{

class MSE
{
public:
    template <typename S, int A, int B>
    const static mat<S, A, B> error(const mat<S, A, B> &output, const mat<S, A, B> &expected)
    {
        return output - expected;
    }

    template <typename S, int A, int B>
    const static vec<S, B> cost(const mat<S, A, B> &output, const mat<S, A, B> &expected)
    {
        auto diff = output - expected;
        return diff.transpose() * diff;
    }
};

} // namespace cost

} // namespace neural