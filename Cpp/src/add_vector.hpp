#pragma once

#include "real.hpp"

class AddVector
{
public:
	static void initialise(Array& matrix, Array& vector, int m, int n);
	static void native(Array& matrix, const Array& vector, int m, int n);
	static void cuda(Array& matrix, const Array& vector, int m, int n);
};