#pragma once

#include <Eigen/Dense>

namespace neural
{

template <typename S, int A>
using vec = Eigen::Matrix<S, A, 1>;

template <typename S, int A, int B>
using mat = Eigen::Matrix<S, A, B>;

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
                          std::make_index_sequence<
                              std::tuple_size<Tuple>::value - 1>());
}

namespace activation
{

struct Linear
{
public:
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> g(const mat<S, A, Batch> &x)
    {
        return x;
    }
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> gprime(
        const mat<S, A, Batch> &x,
        const mat<S, A, Batch> &y)
    {
        return mat<S, A, Batch>::Ones();
    }
};

struct Sigmoid
{

public:
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> g(const mat<S, A, Batch> &x)
    {
        return 1 / (1 + exp(-x.array()));
    }
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> gprime(
        const mat<S, A, Batch> &x,
        const mat<S, A, Batch> &y)
    {
        return y.array() * (1 - y.array());
    }
};

struct Relu
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
        const S operator()(const S &x) const
        {
            if (x > 0)
                return 1;
            else
                return 0;
        }
    };

public:
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> g(const mat<S, A, Batch> &x)
    {
        return x.unaryExpr(CwiseRelu<S>());
    }
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> gprime(
        const mat<S, A, Batch> &x,
        const mat<S, A, Batch> &y)
    {
        return x.unaryExpr(ReluPrime<S>());
    }
};

} // namespace activation

namespace cost
{

struct MSE
{
public:
    template <typename S, int A, int Batch>
    const static mat<S, A, Batch> error(
        const mat<S, A, Batch> &output,
        const mat<S, A, Batch> &expected)
    {
        return output - expected;
    }

    template <typename S, int A, int Batch>
    const static vec<S, Batch> cost(
        const mat<S, A, Batch> &output,
        const mat<S, A, Batch> &expected)
    {
        auto diff = output - expected;
        return diff.colwise().norm();
    }
};

} // namespace cost

} // namespace neural