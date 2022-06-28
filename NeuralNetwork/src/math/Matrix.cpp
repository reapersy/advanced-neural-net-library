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

void Matrix::ZeroOut()
{
	std::fill(m_Matrix.begin(), m_Matrix.end(), 0);
}

std::vector<double> Matrix::GetColumnVector() const
{
#ifdef _DEBUG
	if (m_Columns != 1)
		throw MatrixError("Number of columns has to be 1 in order to make column vector!");
#endif // _DEBUG
	return m_Matrix;
}

void Matrix::SaveMatrix(std::ofstream & outfile) const
{
	outfile.write((char*)(&m_Rows), sizeof(m_Rows));
	outfile.write((char*)(&m_Columns), sizeof(m_Columns));
	outfile.write((char*)&m_Matrix[0], sizeof(double)*m_Rows*m_Columns);
}

double & Matrix::operator()(unsigned int row, unsigned int column)
{
#ifdef _DEBUG
	if ((column + row*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[column + row*m_Columns];
}

const double & Matrix::operator()(unsigned int row, unsigned int column) const
{
#ifdef _DEBUG
	if ((column + row*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[column + row*m_Columns];
}

double & Matrix::operator[](const std::pair<unsigned int, unsigned int>& index)
{
#ifdef _DEBUG
	if ((index.second + index.first*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[index.second + index.first*m_Columns];
}

const double & Matrix::operator[](const std::pair<unsigned int, unsigned int>& index) const
{
#ifdef _DEBUG
	if ((index.second + index.first*m_Columns) >= m_Rows*m_Columns)
		throw MatrixError("Index out of range!");
#endif // _DEBUG
	return m_Matrix[index.second + index.first*m_Columns];
}

Matrix & Matrix::operator+=(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	std::transform(m_Matrix.begin(), m_Matrix.end(), other.m_Matrix.begin(), m_Matrix.begin(), std::plus<double>());
	return *this;
}

Matrix & Matrix::operator+=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x += scalar; });
	return *this;
}

Matrix & Matrix::operator-=(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	std::transform(m_Matrix.begin(), m_Matrix.end(), other.m_Matrix.begin(), m_Matrix.begin(), std::minus<double>());
	return *this;
}

Matrix & Matrix::operator-=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x -= scalar; });
	return *this;
}

Matrix & Matrix::operator*=(double scalar)
{
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x *= scalar; });
	return *this;
}

Matrix & Matrix::operator*=(const Matrix & other)
{
#ifdef _DEBUG
	if (m_Columns != other.m_Rows)
		throw MatrixError("Number of columns of the left matrix has to match number of rows of the right matrix!");
#endif // _DEBUG
	*this = *this * other;
	return *this;
}

Matrix & Matrix::operator/=(double scalar)
{
#ifdef _DEBUG
	if (scalar == 0)
		throw MatrixError("Cannot divide by zero!");
#endif // _DEBUG
	std::for_each(m_Matrix.begin(), m_Matrix.end(), [scalar](double& x) { x /= scalar; });
	return *this;
}

Matrix & Matrix::DotProduct(const Matrix & other)
{
#ifdef _DEBUG
	if (!HasSameDimension(other))
		throw MatrixError("Matrices do not have the same dimension!");
#endif // _DEBUG
	std::transform(m_Matrix.begin(), m_Matrix.end(), other.m_Matrix.begin(), m_Matrix.begin(), std::multiplies<double>());
	return *this;
}

Matrix & Matrix::Transpose()
{
	if (m_Rows != 1 && m_Columns != 1)
	{
		std::vector<double> transposedMatrix(m_Rows*m_Columns);
		for (unsigned int i = 0; i < m_Rows; ++i)
		{
			for (unsigned int j = 0; j < m_Columns; ++j)
			{
				transposedMatrix[i + j*m_Rows] = m_Matrix[j + i*m_Columns];
			}
		}
		m_Matrix = std::move(transposedMatrix);
	}
	std::swap(m_Rows, m_Columns);
	return *this;
}

Matrix Matrix::LoadMatrix(std::ifstream & infile)
{
	unsigned int rows, columns;
	infile.read((char*)&rows, sizeof(rows));
	infile.read((char*)&columns, sizeof(columns));
	Matrix matrix(rows, columns);
	double* m = new double[rows*columns];
	infile.read((char*)m, sizeof(double)*rows*columns);
	std::vector<double> mat(m, m + rows*columns);
	matrix.m_Matrix = std::move(mat);
	delete[] m;
	return matrix;
}

Matrix Matrix::OneHot(unsigned int one, unsigned int size)
{
	Matrix matrix(1, size, 0);
	matrix(0, one) = 1;
	return matrix;
}

Matrix Matrix::DotProduct(const Matrix & left, const Matrix & right)
{
#ifdef _DEBUG
	if (!left.HasSameDimension(right))
		throw MatrixError("Matrices do not have the same