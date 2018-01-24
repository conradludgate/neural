#pragma once

#include <neural/template_helper.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

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

	template<ui n, ui A, ui B, ui... Is>
	vec<get_last<B, Is...>()> process(vec<A> input)
	{
		auto output = relu<B>(std::get<n>(m_weights) * input + std::get<n>(m_biases));		
		if constexpr(sizeof...(Is) != 0)
	    {
			return process<n+1, B, Is...>(output);
		}

		return output;
	}

private:
	template<ui n>
	void random()
	{
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
		std::get<n>(m_weights).setZero();
		std::get<n>(m_biases).setZero();

		if constexpr(sizeof...(Further) != n)
		{
			zero<n+1>();
		}
	}

	// For serialization of the network
	friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        _serialize<Archive, 0>(ar, version);
    }

    template<class Archive, ui n>
    void _serialize(Archive & ar, const unsigned int version)
    {
        auto w = std::get<n>(m_weights);
        for (int j = 0; j < w.cols(); ++j)
        	for (int i = 0; i < w.rows(); ++i)
        		ar & w(i, j);

		auto b = std::get<n>(m_biases);
		for (int i = 0; i < b.rows(); ++i)
			ar & b[i];

		if constexpr(sizeof...(Further) != n)
		{
			_serialize<Archive, n+1>(ar, version);
		}
    }
};

}