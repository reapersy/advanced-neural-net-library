
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

#include "WeightInitializers.h"

namespace nn
{
	namespace initialization
	{
		void HeUniform::Initialize(Matrix& matrix) const
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return (valueDistribution(engine) - 0.5) * factor;
			});
		}
	};
};