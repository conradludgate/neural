#pragma once

#include <Eigen/Dense>
#include <tuple>

// Used a lot...
//using ui = std::uint32_t;

// Shorthand for vectors
template<int A>
using vec = Eigen::Matrix<float, A, 1>;

// Shorthand for matricies
template<int A, int B>
using mat = Eigen::Matrix<float, A, B>;

template<int B, int... Is>
constexpr int get_last()
{
	if constexpr(sizeof...(Is) == 0)
	{
		return B;
	}
	else
	{
		return get_last<Is...>();
	}
}

template <int A, int B>
auto make_weights()
{
	return std::tuple<mat<B, A>>{};
}

template <int A, int B, int C>
auto make_weights()
{
	return std::tuple<mat<B, A>, mat<C, B>>{};
}

template <int A, int B, int C, int D>
auto make_weights()
{
	return std::tuple<mat<B, A>, mat<C, B>, mat<D, C>>{};
}

template <int A, int B, int C, int D, int E, int... Is>
auto make_weights()
{   
	if constexpr(sizeof...(Is) == 0)
	{
		return std::tuple<mat<B, A>, mat<C, B>, mat<D, C>, mat<E, D>>{};
	}
	else
	{
		return std::tuple_cat(make_weights<A, B, C, D, E>(), make_weights<E, Is...>());
	}
}

template <int B>
auto make_biases()
{
	return std::tuple<vec<B>>{};
}

template <int B, int C>
auto make_biases()
{
	return std::tuple<vec<B>, vec<C>>{};
}

template <int B, int C, int D>
auto make_biases()
{
	return std::tuple<vec<B>, vec<C>, vec<D>>{};
}

template <int B, int C, int D, int E, int... Is>
auto make_biases()
{   
	if constexpr(sizeof...(Is) == 0)
	{
		return std::tuple<vec<B>, vec<C>, vec<D>, vec<E>>{};
	}
	else
	{
		return std::tuple_cat(make_biases<B, C, D, E>(), make_biases<Is...>());
	}
}

// ReLU
// template<int B>
// vec<B> g(vec<B> v)
// {
// 	for (int i = 0; i < B; ++i)
// 		if (v[i] < 0)
// 		{
// 			v[i] = 0;
// 		}

// 	return v;
// }

// template<int B>
// void gprime(vec<B>& s, vec<B> v)
// {
// 	for (int i = 0; i < B; ++i)
// 		if (v[i] <= 0)
// 		{
// 			s[i] = 0;
// 		}
// }

// Sigmoid
template<int B>
vec<B> g(vec<B> v)
{
	for (int i = 0; i < B; ++i)
		v[i] = 1 / (1 - exp(-v[i]));

	return v;
}

template<int B>
void gprime(vec<B>& s, vec<B> v)
{
	for (int i = 0; i < B; ++i)
		s[i] *= (1 - v[i]) * v[i];
}