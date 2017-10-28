#pragma once

#include "neural/net.hpp"
#include <cmath>

namespace neural
{

struct TestData
{
	std::vector<float> input;
	std::vector<float> expected;
};

class Trainer: public Net
{
public:
	virtual float costf(std::vector<float> output, std::vector<float> expected) = 0;
	//virtual TestData<Tin, Tout> dataf() = 0;
	virtual std::vector<TestData> data_set() = 0;
	//virtual Tout create_out() = 0;

	void train(float tol, float step)
	{
		auto test_data = data_set();
		auto cost = test(test_data);

		while (cost > tol)
		{
			auto delta = cost * step;

			std::vector<float> new_weights(m_weights.size());

			// auto j = new_weights.begin();
			// for (auto i = m_weights.begin(); i != m_weights.end(); ++i)
			// {
			// 	*i += delta;
			// 	*(j++) = test(test_data);
			// 	*i -= delta;
			// }

			for (int i = 0; i < m_weights.size(); ++i)
			{
				m_weights[i] += delta;
				new_weights[i] = m_weights[i] + (cost - test(test_data)) * delta;
				m_weights[i] -= delta;
			}

			m_weights = new_weights;

			cost = test(test_data);
			test_data = data_set();
		}
	}

	float test(std::vector<TestData>& test_data)
	{
		double total = 0;

		for (auto i = test_data.begin(); i != test_data.end(); ++i)
		{
			total += std::abs(costf(process((*i).input), (*i).expected));
		}

		return total / test_data.size();
	}
};

}