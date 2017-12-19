#pragma once

#include <Eigen/Dense>
#include <tuple>

using ui = std::uint32_t;

template<ui A>
using vec = Eigen::Matrix<float, A, 1>;

template<ui A, ui B>
using mat = Eigen::Matrix<float, A, B>;

template<ui A, ui... Is>
constexpr ui get_last()
{
	if constexpr(sizeof...(Is) == 0)
    {
        return A;
    }
    else
    {
        return get_last<Is...>();
    }
}

template<ui... Is>
ui count()
{
	return sizeof...(Is);
}

template <ui A, ui B, ui... Is>
auto make_weights()
{   
    if constexpr(sizeof...(Is) == 0)
    {
        return std::tuple<mat<B, A>>{};
    }
    else
    {
        return std::tuple_cat(make_weights<A, B>(), 
        	make_weights<B, Is...>());
    }
}

template <ui B, ui... Is>
auto make_biases()
{
	if constexpr(sizeof...(Is) == 0)
	{
		return std::tuple<vec<B>>{};
	}
	else
	{
		return std::tuple_cat(make_biases<B>(), make_biases<Is...>());
	}
}