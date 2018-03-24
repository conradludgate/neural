#pragma once

#include <neural/template_helper.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

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

	void Save(const char* filename)
	{
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oa(ofs);
    	oa << *this;
	}

	void Load(const char* filename)
	{
		std::ifstream ifs(filename);
		boost::archive::binary_iarchive ia(ifs);
    	ia >> *this;
	}

	vec<last> process(vec<First> input)
	{
		return process<0, First, Second, Further...>(input);
	}

	
protected:
	decltype(make_weights<First, Second, Further...>()) m_weights;
	decltype(make_biases<Second, Further...>()) m_biases;

	template<int n, int A, int B, int... Is>
	vec<get_last<B, Is...>()> process(vec<A> input)
	{
		if constexpr(sizeof...(Is) == 0)
	    {
	        return g<B>(std::get<n>(m_weights) * input + std::get<n>(m_biases));
	    }
	    else
	    {
			return process<n+1, B, Is...>(process<n, A, B>(input));
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

	// For serialization of the network
	friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        _serialize<Archive, 0>(ar, version);
    }

    template<class Archive, int n>
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