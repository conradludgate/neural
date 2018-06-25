#pragma once

#include <iostream>
#include <type_traits>

#include "neural/util.hpp"

namespace neural
{

// template<int I, int O>
// class ANN
// {
// public:
// 	virtual void random() = 0;
// 	virtual vec<O>& feedforward(vec<I>&) = 0;
// 	virtual vec<I>& feedbackward(vec<O>&) = 0;
// 	virtual std::ostream& operator<<(std::ostream&) = 0;
// 	virtual std::istream& operator>>(std::istream&) = 0;
// };

template<typename A, typename... Bs>
class Net //: public ANN
{
private:
	std::tuple<A, Bs...> layers;

public:

	// Input and output
	static const int I = A::I;
	static const int O = std::tuple_element<sizeof...(Bs), decltype(layers)>::type::O;

	void random() { random<0>(); }
	std::ostream& operator<<(std::ostream& os) { return save<0>(os); } 
	std::istream& operator>>(std::istream& is) { return load<0>(is); } 

	void train(Scalar lr, vec<I> input, vec<O> target)
	{
		feedbackward(lr, (target - feedforward(input)));
	}

	vec<O> feedforward(const vec<I>& input)
	{
		return feedforward<0, I>(input);
	}

	vec<I> feedbackward(Scalar lr, const vec<O>& error)
	{
		return feedbackward<sizeof...(Bs), O>(lr, error);
	}

private:

	template<int n>
	void random()
	{
		std::get<n>(layers).random();
		if constexpr(sizeof...(Bs) != n)
			random<n+1>();
	}

	template<int n>
	std::ostream& save(std::ostream& os)
	{
		std::get<n>(layers).save(os);
		if constexpr(n != sizeof...(Bs))
		{
			save<n+1>(os);
		}
		return os;
	}

	template<int n>
	std::istream& load(std::istream& is)
	{
		std::get<n>(layers).load(is);
		if constexpr(n != sizeof...(Bs))
		{
			load<n+1>(is);
		}
		return is;
	}

	template<int n, int inputs>
	vec<O> feedforward(vec<inputs> input)
	{
		auto& l = std::get<n>(layers);
		if constexpr(n == sizeof...(Bs))
		{
			return l.feedforward(input);
		}
		else
		{
			return feedforward<n+1, 
				std::tuple_element<n, decltype(layers)>::type::O
			>(l.feedforward(input));
		}
	}

	template<int n, int outputs>
	vec<I> feedbackward(Scalar lr, const vec<outputs>& error)
	{
		auto& l = std::get<n>(layers);
		if constexpr(n == 0)
		{
			return l.feedbackward(lr * error);
		}
		else
		{
			return feedbackward<n-1, 
				std::tuple_element<n, decltype(layers)>::type::I
			>(lr, l.feedbackward(lr * error));
		}
	}
};

}