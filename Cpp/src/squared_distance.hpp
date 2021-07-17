#pragma once

#include "real.hpp"

class SquaredDistance
{
public:
	static void initialise(Array& mSquaredDistances, Array& mCoordinates, int c, int n);
	static void native(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
	static void cuda(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
	static void cuda_shared_memory(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
	static void cuda_float2(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
	static void cuda_constant(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
	static void cuda_local_memory(Array& mSquaredDistances, const Array& mCoordinates, int c, int n);
};
