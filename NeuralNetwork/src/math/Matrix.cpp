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
		std::fill(m_Matrix.begin(), m_Matrix.end(), initValue);
}

Matrix::Matrix(const Matrix & matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(matrix.m_Matrix)
{
}

Matrix::Matrix(Matrix && matrix) : m_Rows(matrix.m_Rows), m_Columns(matrix.m_Columns), m_Matrix(std::move(matrix.m_Matrix))
{

}

Matrix::Matrix(const std::vector<double>& data) : m_Rows(data.size()), m_Columns(1), m_Matrix(data)
{
}

#ifdef _DEBUG
Matrix::Matrix(const std::vector<std::vector<double>>& matrix) : m_Rows(matrix.size()), m_Columns(matrix[0].size()), m_Matrix(matrix.size()*matrix[0].size())
{
	for (unsigned int i = 0; i < m_Rows; ++i)
	{
		for (unsigned int j = 0; j < m_Columns; ++j)
		{
			(*this)(i, j) = matrix[i][j];
		}
	}
}
#endif // _DEBUG


Matrix & Matrix::operator=(const Matrix & matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns;
	m_Matrix = matrix.m_Matrix;
	return *this;
}

Matrix & Matrix::operator=(Matrix && matrix)
{
	m_Rows = matrix.m_Rows; m_Columns = matrix.m_Columns; m_Matrix = std::move(matrix.m_Matrix);
	return *this;
}

Matrix::~Matrix()
{
}

double Matrix::Sum() const
{
	double sum = 0.0;
	return std::accumulate(m_Matrix.begin(), m_Matrix.end(), sum);
}

void Matrix::Randomize(double min, double max)
{

	std::random_device randomDevice;
	std::mt19937 engine(randomDevice());
	std::uniform_real_distribution<double> valueDistribution(min, max);
	for (unsigned int i = 0; i < m_Rows*m_Columns; ++i)
	{
		m_Matrix[i] = valueDistribution(engine);
	}
}

v