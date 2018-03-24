#pragma once

#include <neural/template_helper.hpp>

#include <fstream>

namespace neural
{

template<int First, int Second, int... Further>
class Net
{

static const int last = get_last<Second, Further...>();

public:
	void Zero()
	{
		zero<0>();
	}

	void Random()
	{
		random<0>();
	}

	void Constant(float c)
	{
		constant<0>(c);
	}

	vec<last> predict(vec<First> input)
	{
		return predict<0, First, Second, Further...>(input);
	}

protected:
	decltype(make_weights<First, Second, Further...>()) m_weights;
	decltype(make_biases<Second, Further...>()) m_biases;

	template<int n, int A, int B, int... Is>
	vec<get_last<B, Is...>()> predict(vec<A> input)
	{
		if constexpr(sizeof...(Is) == 0)
	    {
	        return g<B>(std::get<n>(m_weights) * input + std::get<n>(m_biases));
	    }
	    else
	    {
			return predict<n+1, B, Is...>(predict<n, A, B>(input));
		}
	}

private:
	template<int n>
	void random()
	{
		std::get<n>(m_weights).setRandom();
		std::get<n>(m_biases).setRandom();

		if constexpr(sizeof...(Further) != n)
		{
			random<n+1>();
		}
	}

	template<int n>
	void zero()
	{
		std::get<n>(m_weights).setZero();
		std::get<n>(m_biases).setZero();

		if constexpr(sizeof...(Further) != n)
		{
			zero<n+1>();
		}
	}

	template<int n>
	void constant(float c)
	{
		std::get<n>(m_weights).setConstant(c);
		std::get<n>(m_biases).setConstant(c);

		if constexpr(sizeof...(Further) != n)
		{
			zero<n+1>();
		}
	}
};

}