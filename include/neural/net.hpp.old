#pragma once

#include <vector>
#include <iostream>

#include <type_traits>

namespace neural 
{

class Net
{
public:
	// Net(std::vector<unsigned int> nodes)
	// {
	// 	setSize(nodes);
	// }

	void setSize(std::vector<unsigned int> nodes)
	{
		m_nodes = nodes;

		unsigned int weights = 0;
		for (auto i = nodes.begin(); i != nodes.end() - 1; ++i)
		{
			weights += (*i) * (*(i + 1));
		}

		m_weights = std::vector<float>(weights);
	}

	std::vector<float> process(std::vector<float> in)
	{
		if (in.size() < m_nodes[0])
			in.resize(m_nodes[0]);

		int steps = m_nodes.size() - 1;
		auto i_weight = m_weights.begin();

		std::vector<float> out;

		for (int step = 0; step < steps; ++step)
		{
			out = std::vector<float>(m_nodes[step + 1]);

			step_once<std::vector<float>, std::vector<float>>(in, out, i_weight);

			in = out;
		}

		return out;
	}

	template<typename Tin>
	std::vector<float> process(Tin input)
	{
		auto i_weight = m_weights.begin();
		std::vector<float> out(m_nodes[1]);

		step_once<Tin, std::vector<float>>(input, out, i_weight);

		int steps = m_nodes.size() - 1;

		auto in = out;
		for (int step = 1; step < steps; ++step)
		{
			out = std::vector<float>(m_nodes[step + 1]);

			step_once<std::vector<float>, std::vector<float>>(in, out, i_weight);

			in = out;
		}

		return out;
	}

	template<typename Tin>
	std::vector<float> process(std::vector<Tin> input)
	{
		auto i_weight = m_weights.begin();
		std::vector<float> out(m_nodes[1]);

		step_once<std::vector<Tin>, std::vector<float>>(input, out, i_weight);

		int steps = m_nodes.size() - 1;

		auto in = out;
		for (int step = 1; step < steps; ++step)
		{
			out = std::vector<float>(m_nodes[step + 1]);

			step_once<std::vector<float>, std::vector<float>>(in, out, i_weight);

			in = out;
		}

		return out;
	}

	template<typename Tin, typename Tout>
	void process(Tin input, Tout& output)
	{
		auto i_weight = m_weights.begin();

		int steps = m_nodes.size() - 2;
		//Tout final_out(m_nodes[steps + 1]);

		if (steps == 0)
		{
			step_once<Tin, Tout>(input, output, i_weight);
			return;
			//return output;
		}

		std::vector<float> out(m_nodes[1]);

		step_once<Tin, std::vector<float>>(input, out, i_weight);

		auto in = out;
		for (int step = 1; step < steps; ++step)
		{
			out = std::vector<float>(m_nodes[step + 1]);

			step_once<std::vector<float>, std::vector<float>>(in, out, i_weight);

			in = out;
		}

		step_once<std::vector<float>, Tout>(in, output, i_weight);

		//return final_out;
	}

	template<typename Tin, typename Tout>
	void process(std::vector<Tin> input, Tout& output)
	{
		auto i_weight = m_weights.begin();

		int steps = m_nodes.size() - 2;
		//Tout final_out(m_nodes[steps + 1]);

		if (steps == 0)
		{
			step_once<std::vector<Tin>, Tout>(input, output, i_weight);
			return;
			//return output;
		}

		std::vector<float> out(m_nodes[1]);

		step_once<std::vector<Tin>, std::vector<float>>(input, out, i_weight);

		auto in = out;
		for (int step = 1; step < steps; ++step)
		{
			out = std::vector<float>(m_nodes[step + 1]);

			step_once<std::vector<float>, std::vector<float>>(in, out, i_weight);

			in = out;
		}

		step_once<std::vector<float>, Tout>(in, output, i_weight);

		//return final_out;
	}

protected:
	// GPU Optimise?
	template<typename Tin, typename Tout>
	void step_once(Tin& in, Tout& out, std::vector<float>::iterator i_weight)
	{
		for (auto i_out = out.begin(); i_out != out.end(); ++i_out)
		{
			for (auto i_in = in.begin(); i_in != in.end(); ++i_in)
			{
				*i_out += (*i_in) * (*(i_weight++));
			}
		}
	}

	std::vector<unsigned int> m_nodes;
	std::vector<float> m_weights;
};

}