#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage
	// TODO: 'nn' namespace sam stavio kao placeholder dok ne smislimo nesto bolje
	nn::NeuralNetwork model(2, {
		nn::Layer(2, 4, nn::activation::SIGMOID),
		nn::Layer(4, 4, nn::activation::SIGMOID),
		nn::Layer(4, 1, nn::activation::SIGMOID)
	}, nn::initialization::XAVIER_NORMAL, nn::loss::QUADRATIC);

	// Getting the data
	std::vector<nn::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

	// Training
	unsigned int epochs = 1000;
	unsigned in