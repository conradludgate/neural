#pragma once

namespace neural
{

template <typename Scalar, typename Cost, int O>
class CostLayer
{

public:
    static const int Outputs = O;

    template <int Batch>
    mat<Scalar, Outputs, Batch> feedforward_backward(Scalar learning_rate, mat<Scalar, Outputs, Batch> input, mat<Scalar, Outputs, Batch> expected)
    {
        return learning_rate * Cost::template cost<Scalar, Outputs, Batch>(input, expected);
    }
};

} // namespace neural