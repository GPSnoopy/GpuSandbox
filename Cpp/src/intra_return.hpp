#pragma once

#include "real.hpp"

class IntraReturn
{
public:
	static void initialise(Array& mIntraReturn, Array& vClose, Array& vIsAlive, Array& vIsValidDay, int m, int n);
	static void native(Array& mIntraReturn, const Array& vClose, const Array& vIsAlive, const Array& vIsValidDay, int m, int n);
	static void cuda(Array& mIntraReturn, const Array& vClose, const Array& vIsAlive, const Array& vIsValidDay, int m, int n);
};