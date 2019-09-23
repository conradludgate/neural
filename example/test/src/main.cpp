#include <iostream>

#include <cstdlib>
#include <csignal>

#include "neural/net.hpp"
#include "neural/layer.hpp"
#include "neural/serial.hpp"

template <int A>
using vec = neural::vec<float, A>;

template <int A, int B>
using mat = neural::mat<float, A, B>;

typedef neural::Net<
    float,                           // Scalar type
    neural::cost::MSE,               // Cost function
    neural::LinearLayer<float, 2, 6, // First Layer
                        neural::activation::Linear>,
    neural::LinearLayer<float, 6, 1, // Second Layer
                        neural::activation::Linear>>
    NN;

template <int Batch>
struct Sum
{
    mat<NN::Inputs, Batch> input;
    mat<NN::Outputs, Batch> expected;

    Sum()
    {
        input = mat<NN::Inputs, Batch>::Random();
        expected = input.colwise().sum();
    }
};

const int train_iter = 1000;
const int training_batch_size = 512;
const int test_iter = 100;
const int testing_batch_size = 256;

NN test;

void on_exit()
{
    std::cout << "Saving "
              << neural::serial::save(test, "networks", "test") << std::endl;
}

void sig_handler(int s)
{
    on_exit();
    exit(0);
}

int main()
{
    srand(time(nullptr));

    auto filename = neural::serial::load(test, "networks", "test");
    if (filename.length())
        std::cout << "Loading " << filename << std::endl;
    else
    {
        test.random();
        std::cout << "Loading random data" << std::endl;
    }

    signal(SIGINT, sig_handler);

    // Set the learning rate for the network (default 0.1)
    float lr = 0.2;

    // Train until satisfied.
    float cost = 0.5;
    int epochs = 0;
    while (cost > 1E-6)
    {
        epochs++;

        // Train batch
        for (int i = 0; i < train_iter; i++)
        {
            auto training_data = Sum<training_batch_size>();
            test.train(lr, training_data.input, training_data.expected);
        }

        // Test batch
        cost = 0;
        for (int i = 0; i < test_iter; i++)
        {
            auto testing_data = Sum<testing_batch_size>();
            auto output = test.predict(testing_data.input);
            auto c = NN::Cost::cost(output, testing_data.expected).sum();
            cost += c / testing_batch_size / test_iter;
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

    on_exit();
}