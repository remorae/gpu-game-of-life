#include "Kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int sharedPos(int tx, int dx)
{
	return tx + 1 + dx;
}

__device__ int globalPos(int ix, int width, int dx)
{
	return (ix + dx + width) % width;
}

__device__ int sharedIndex(int tx, int ty, int sharedWidth, int dx, int dy)
{
	return sharedPos(ty, dy) * sharedWidth + sharedPos(tx, dx);
}

__device__ int globalIndex(int ix, int iy, int width, int height, int dx, int dy)
{
	return globalPos(iy, height, dy) * width + globalPos(ix, width, dx);
}

__device__ int onBorder(int threadPos, int globalPos, int blockEdge, int gridEdge)
{
	return (threadPos == blockEdge - 1 || globalPos == gridEdge - 1);
}

__global__ void updateGrid(unsigned char* destGrid, unsigned char* srcGrid, const int width, const int height, const int offsetX, const int offsetY)
{
	extern __shared__ int s_data[];
	const int sharedMemDim_x = blockDim.x + 2;

	const unsigned int iy = offsetY + blockIdx.y * blockDim.y + threadIdx.y; //row index
	const unsigned int ix = offsetX + blockIdx.x * blockDim.x + threadIdx.x; //column index
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	if (iy >= height || ix >= width)
		return;

	// Shared memory
	if (ty == 0)
	{
		// N
		s_data[sharedIndex(tx, ty, sharedMemDim_x, 0, -1)] = srcGrid[globalIndex(ix, iy, width, height, 0, -1)];
		if (tx == 0)
		{
			// NW
			s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, -1)] = srcGrid[globalIndex(ix, iy, width, height, -1, -1)];
		}
		if (onBorder(tx, ix, blockDim.x, width))
		{
			// NE
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, -1)] = srcGrid[globalIndex(ix, iy, width, height, 1, -1)];
		}
	}
	else if (onBorder(ty, iy, blockDim.y, height))
	{
		// S
		s_data[sharedIndex(tx, ty, sharedMemDim_x, 0, 1)] = srcGrid[globalIndex(ix, iy, width, height, 0, 1)];
		if (tx == 0)
		{
			// SW
			s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, 1)] = srcGrid[globalIndex(ix, iy, width, height, -1, 1)];
		}
		if (onBorder(tx, ix, blockDim.x, width))
		{
			// SE
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, 1)] = srcGrid[globalIndex(ix, iy, width, height, 1, 1)];
		}
	}

	if (tx == 0)
	{
		// W
		s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, 0)] = srcGrid[globalIndex(ix, iy, width, height, -1, 0)];
	}
	if (onBorder(tx, ix, blockDim.x, width))
	{
		// E
		s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, 0)] = srcGrid[globalIndex(ix, iy, width, height, 1, 0)];
	}
	
	// Itself
	const int currentSharedIndex = sharedIndex(tx, ty, sharedMemDim_x, 0, 0);
	const int currentGlobalIndex = globalIndex(ix, iy, width, height, 0, 0);
	s_data[currentSharedIndex] = srcGrid[currentGlobalIndex];

	__syncthreads();

	const unsigned char currentCell = s_data[currentSharedIndex];

	const int aliveNeighbors = (	
			s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, -1)] + //NW
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 0, -1)] + //N
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, -1)] + //NE
			s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, 0)] + //W
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, 0)] + //E
			s_data[sharedIndex(tx, ty, sharedMemDim_x, -1, 1)] + //SW
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 0, 1)] + //S
			s_data[sharedIndex(tx, ty, sharedMemDim_x, 1, 1)] //SE   
						);
	
	if (currentCell > 0 && aliveNeighbors != 2 && aliveNeighbors != 3)
		destGrid[currentGlobalIndex] = 0;
	else if (currentCell == 0 && aliveNeighbors == 3)
		destGrid[currentGlobalIndex] = 1;
}
