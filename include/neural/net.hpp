#pragma once

#include <neural/template_helper.hpp>

namespace neural
{

template<ui First, ui Second, ui... Further>
class Net
{

static const ui last = get_last<Second, Further...>();

public:
	void Zero()
	{
		zero<0>();
	}

	void Random()
	{
		random<0>();
	}

	vec<last> process(vec<First> input)
	{
		return process<0, First, Second, Further...>(input);
	}

protected:
	decltype(make_weights<First, Second, Further...>()) m_weights;
	decltype(make_biases<Second, Further...>()) m_biases;

	template<ui B>
	vec<B> relu(vec<B> v)
	{
		for (int i = 0; i < B; ++i)
			if (v[i] < 0)
				v[i] = 0;

		return v;
	}

	template<ui n, ui A, ui B, ui... Is>
	vec<B> process(vec<A> input)
	{
		if constexpr(sizeof...(Is) == 0)
	    {
	        return relu<B>(std::get<n>(m_weights) * input + std::get<n>(m_biases));
	        //return std::get<n>(m_weights) * input + std::get<n>(m_biases);
	    }
	    else
	    {
			return process<n+1, B, Is...>(process<n, A, B>());
		}
	}

private:
	template<ui n>
	void random()
	{
		//std::get<n>(m_weights) = Eigen::Matrix<float, B, A>::Random();
		std::get<n>(m_weights).setRandom();
		std::get<n>(m_biases).setRandom();

		if constexpr(sizeof...(Further) != n)
		{
			random<n+1>();
		}
	}

	template<ui n>
	void zero()
	{
		//std::get<n>(m_weights) = Eigen::Matrix<float, B, A>::Zero();
		std::get<n>(m_weights).setZero();
		std::get<n>(m_biases).setZero();

		if constexpr(sizeof...(Further) != n)
		{
			zero<n+1>();
		}
	}
};

}