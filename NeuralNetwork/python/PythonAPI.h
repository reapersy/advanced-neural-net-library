
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

#pragma once
#include "../NeuralNetwork.h"

#define NN_API extern "C" __declspec(dllexport)

typedef struct sgd
{
	double lr;
} SGD;

typedef struct momentum
{
	double lr;
	double moment;
} Momentum;

typedef struct nesterov
{
	double lr;
	double moment;
} Nesterov;

typedef struct adagrad
{
	double lr;
} Adagrad;

typedef struct rmsprop
{
	double lr;
	double beta;
} RMSProp;

typedef struct adadelta
{
	double lr;
	double beta;
} Adadelta;

typedef struct adam
{
	double lr;
	double beta1;
	double beta2;
} Adam;

typedef struct nadam
{
	double lr;
	double beta1;
	double beta2;
} Nadam;

typedef struct adamax
{
	double lr;
	double beta1;
	double beta2;
} Adamax;

typedef struct amsgrad
{
	double lr;
	double beta1;
	double beta2;
} AMSGrad;

typedef struct adabound
{
	double lr;
	double beta1;
	double beta2;
	double final_lr;
	double gamma;
} Adabound;

typedef struct amsbound
{
	double lr;
	double beta1;
	double beta2;
	double final_lr;
	double gamma;
} AMSBound;

typedef struct output
{
	double value;
	unsigned int argmax;
} Output;

typedef struct dense
{
	unsigned int neurons;
	unsigned int activation_function;
	unsigned int inputs;
} Dense;

typedef struct model
{
	std::unique_ptr<nn::NeuralNetwork> net;
	std::unique_ptr<nn::optimizer::Optimizer> optimizer;
	std::vector<nn::Layer> layers;
	unsigned int inputSize;
	unsigned int outputSize;
	unsigned int regularizerType;
} Model;

static Model model;
static std::vector<nn::TrainingData> trainingData;
static const double default_lr = 0.01;

void create_optimizer(nn::optimizer::Type type, void* ptr = NULL)
{
	switch (type)
	{
	case nn::optimizer::Type::GRADIENT_DESCENT:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::GradientDescent>(default_lr);
		else
		{
			SGD* sgd = (SGD*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::GradientDescent>(sgd->lr);
		}
		break;
	case nn::optimizer::Type::MOMENTUM:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Momentum>(default_lr);
		else
		{
			Momentum* momentum = (Momentum*)ptr;
			model.optimizer = std::make_unique <nn::optimizer::Momentum>(momentum->lr, momentum->moment);
		}
		break;
	case nn::optimizer::Type::NESTEROV:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Nesterov>(default_lr);
		else
		{
			Nesterov* nesterov = (Nesterov*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Nesterov>(nesterov->lr, nesterov->moment);
		}
		break;
	case nn::optimizer::Type::ADAGRAD:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Adagrad>(default_lr);
		else
		{
			Adagrad* adagrad = (Adagrad*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Adagrad>(adagrad->lr);
		}