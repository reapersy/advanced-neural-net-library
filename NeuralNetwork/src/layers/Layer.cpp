
﻿/*
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

#include "Layer.h"

namespace nn
{
	Matrix Layer::UpdateActivation(const Matrix & input)
	{
		WeightedSum = WeightMatrix*input + BiasMatrix;
		Activation = WeightedSum;
		Activation = ActivationFunction->Function(Activation);
		return Activation;
	}

	Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, nn::activation::Type activationFunction)
		: WeightMatrix(outputNeurons, inputNeurons),
		BiasMatrix(outputNeurons, 1),
		Activation(outputNeurons, 1),
		ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction)),
		WeightedSum(outputNeurons, 1)
	{

	}

	Layer::Layer(Layer && layer) noexcept : WeightMatrix(std::move(layer.WeightMatrix)), BiasMatrix(std::move(layer.BiasMatrix)),
		Activation(std::move(layer.Activation)), WeightedSum(std::move(layer.WeightedSum)), ActivationFunction(std::move(layer.ActivationFunction))
	{