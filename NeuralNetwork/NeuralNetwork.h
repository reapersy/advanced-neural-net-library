/*
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
#include <vector>
#include <unordered_map>
#include "src/layers/Layer.h"
#include "src/optimizers/Optimizers.h"
#include "src/initializers/WeightInitializers.h"
#include "src/losses/LossFunctions.h"
#include "src/regularizers/Regularizers.h"

#ifdef _WINDLL // .dll or .lib
#define PYTHON_API
#