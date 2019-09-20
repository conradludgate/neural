#pragma once

#include <Eigen/Dense>
#include <functional>

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
public:
    template <typename S, int A, int B>
    static mat<S, A, B> g(const mat<S, A, B> &v)
    {
        return v.unaryExpr(std::ref(Sigmoid::sigmoid<S>));
    }
    template <typename S, int A, int B>
    static mat<S, A, B> gprime(const mat<S, A, B> &error, const mat<S, A, B> &output)
    {
        return error.binaryExpr(output, std::ref(Sigmoid::sigmoid_prime<S>));
    }

    template <typename S>
    static S sigmoid(const S &x)
    {
        return 1 / (exp(-x) + 1);
    }

    template <typename S>
    static S sigmoid_prime(const S &a, const S &b)
    {
        return a * (1 - b) * b;
    }
};

class Relu
{
public:
    template <typename S, int A, int B>
    static mat<S, A, B> g(const mat<S, A, B> &v)
    {
        return v.unaryExpr(std::ref(Relu::relu<S>));
    }
    template <typename S, int A, int B>
    static mat<S, A, B> gprime(const mat<S, A, B> &error, const mat<S, A, B> &output)
    {
        return error.binaryExpr(output, std::ref(Relu::relu_prime<S>));
    }

    template <typename S>
    static S relu(const S &x)
    {
        if (x > 0)
            return x;
        else
            return 0;
    }

    template <typename S>
    static S relu_prime(const S &a, const S &b)
    {
        if (b > 0)
            return a;
        else
            return 0;
    }
};

} // namespace activation

namespace cost
{

class MSE
{
public:
    template <typename S, int A, int B>
    static mat<S, A, B> cost(mat<S, A, B> output, mat<S, A, B> expected)
    {
        return output - expected;
    }
};

} // namespace cost

} // namespace neural