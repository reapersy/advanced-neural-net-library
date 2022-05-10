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

#include "Matrix.h"
#include <random>
#include <numeric>
#include <functional>
#include <cstdlib>

#ifdef _DEBUG
#define LOG(x) std::cout << x << std::endl
#endif // _DEBUG


Matrix::Matrix() : m_Rows(0), m_Columns(0), m_Matrix()
{
}

Matrix::Matrix(unsigned int rows, unsigned int columns, double initValue) : m_Rows(rows), m_Columns(columns), m_Matrix(rows*columns)
{
	if (initValue == -1)
		Randomize();
	else
		std::fill(m_Matrix.begin