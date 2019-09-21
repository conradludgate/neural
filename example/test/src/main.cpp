#include <iostream>

#include <cstdlib>
#include <ctime>

#include "neural/net.hpp"
#include "neural/layer.hpp"

template <int A>
using vec = neural::vec<float, A>;

template <int A, int B>
using mat = neural::mat<float, A, B>;

typedef neural::Net<
    float,                                                          // Scalar type
    neural::cost::MSE,                                              // Cost function
    neural::LinearLayer<float, 10, 10, neural::activation::Linear>> // First Layer
    // neural::LinearLayer<float, 10, 1, neural::activation::Linear>>  // Second Layer
    NN;

template <int Batch>
struct Sum
{
    mat<10, Batch> input;
    mat<10, Batch> expected;

    Sum()
    {
        input = mat<10, Batch>::Random();
        expected = input;
    }
};

const int train_iter = 100;
const int training_batch_size = 512;
const int test_iter = 10;
const int testing_batch_size = 256;

int main()
{
    srand(time(nullptr));

    // Create and initialise the network
    NN test;
    test.random();

    // Set the learning rate for the network (default 0.1)
    float lr = 0.2;

    // Train until satisfied.
    for (int epochs = 0; epochs < 100; epochs++)
    {
        for (int i = 0; i < train_iter; i++)
        {
            auto training_data = Sum<training_batch_size>();
            test.train(lr, training_data.input, training_data.expected);
        }

        float cost = 0;
        for (int i = 0; i < test_iter; i++)
        {
            auto testing_data = Sum<testing_batch_size>();
            auto output = test.predict(testing_data.input);
            cost += NN::Cost::cost(output, testing_data.expected).sum() / testing_batch_size / test_iter;
        }

        // Output each step of our training, so we know it's still running
        // And to gauge how good our learning rate is
        std::cout << "Trained for " << epochs << " epochs. "
                  << "Cost        " << cost << std::endl;
    }

    auto testing_data = Sum<8>();
    auto outputs = test.predict(testing_data.input);
    std::cout << "Inputs" << std::endl
              << testing_data.input << std::endl
              << "Outputs" << std::endl
              << outputs << std::endl
              << "Target" << std::endl
              << testing_data.expected << std::endl;
}