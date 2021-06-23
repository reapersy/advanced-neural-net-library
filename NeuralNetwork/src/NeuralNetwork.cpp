
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

#include "../NeuralNetwork.h"

namespace nn
{
	NeuralNetwork::NeuralNetwork(unsigned int inputSize, std::vector<Layer>&& layers, initialization::Type initializer, loss::Type lossFunction)
		: m_InputSize(inputSize), m_Layers(layers), m_WeightInitializer(WeightInitializerFactory::BuildWeightInitializer(initializer)), m_LossFunction(LossFunctionFactory::BuildLossFunction(lossFunction))
	{
		if (m_WeightInitializer != nullptr)
			std::for_each(m_Layers.begin(), m_Layers.end(), [wi = m_WeightInitializer](Layer& layer) { layer.Initialize(wi); });
	}

	NeuralNetwork & NeuralNetwork::operator=(NeuralNetwork && net)
	{
		m_Layers = std::move(net.m_Layers);