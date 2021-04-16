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
	std::vector