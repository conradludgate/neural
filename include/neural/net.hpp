#pragma once

#include <Eigen/Dense>
#include <tuple>

namespace neural
{

using ui = std::uint32_t;

template <ui A, ui B, ui... Is>
auto make_matrix_tuple()
{   
    if constexpr(sizeof...(Is) == 0)
    {
        return std::tuple<Eigen::Matrix<float, B, A>>{};
    }
    else
    {
        return std::tuple_cat(make_matrix_tuple<B, A>(), 
                            make_matrix_tuple<B, Is...>());
    }
}

template <ui B, ui... Is>
auto make_vector_return()
{
	if constexpr(sizeof...(Is) == 0)
    {
        return Eigen::Matrix<float, B, 1>{};
    }
    else
    {
		return make_vector_return<Is...>();
	}
}

// Add scalar type?
template<ui First, ui Second, ui... Further>
class Net
{
public:
	Net()
	{
		init<0, First, Second, Further...>();
	}

	decltype(make_vector_return<Second, Further...>()) process(Eigen::Matrix<float, First, 1> input)
	{
		return process<0, First, Second, Further...>(input);
	}

private:	
	template<ui n, ui A, ui B, ui... Is>
	void init()
	{
		std::get<n>(m_weights) = Eigen::Matrix<float, B, A>::Zero();

		if constexpr(sizeof...(Is) != 0)
		{
			init<n+1, B, Is...>();
		}
	}

	template<ui n, ui A, ui B, ui... Is>
	Eigen::Matrix<float, B, 1> process(Eigen::Matrix<float, A, 1> input)
	{
		if constexpr(sizeof...(Is) == 0)
	    {
	        return std::get<n>(m_weights) * input;
	    }
	    else
	    {
			return process<n+1, B, Is...>(std::get<n>(m_weights) * input);
		}
	}

	decltype(make_matrix_tuple<First, Second, Further...>()) m_weights;
};

}