#include "Kernel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void updateGrid(unsigned char* grid, const int width, const int height, const int offsetX, const int offsetY)
{
	extern __shared__ int s_data[];
	const int sharedMemDim_x = blockDim.x + 2;

	const unsigned int iy = offsetX + blockIdx.y * blockDim.y + threadIdx.y; //row index
	const unsigned int ix = offsetY + blockIdx.x * blockDim.x + threadIdx.x; //column index
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	if (iy >= height || ix >= width)
		return;

	// Shared memory
	if (ty == 0)
	{
		// N
		s_data[ty * sharedMemDim_x + tx + 1] = grid[((iy - 1 + height) % height) * width + ix];
		if (tx == 0)
		{
			// NW
			s_data[ty * sharedMemDim_x + tx] = grid[((iy - 1 + height) % height) * width + (ix - 1 + width) % width];
		}
		else if (tx == blockDim.x - 1)
		{
			// NE
			s_data[ty * sharedMemDim_x + tx + 2] = grid[((iy - 1 + height) % height) * width + (ix + 1) % width];
		}
	}
	else if (ty == blockDim.y - 1)
	{
		// S
		s_data[(ty + 2) * sharedMemDim_x + tx + 1] = grid[((iy + 1) % height) * width + ix];
		if (tx == 0)
		{
			// SW
			s_data[(ty + 2) * sharedMemDim_x + tx] = grid[((iy + 1) % height) * width + (ix - 1 + width) % width];
		}
		else if (tx == blockDim.x - 1)
		{
			// SE
			s_data[(ty + 2) * sharedMemDim_x + tx + 2] = grid[((iy + 1) % height) * width + (ix + 1) % width];
		}
	}

	if (tx == 0)
	{
		// W
		s_data[(ty + 1) * sharedMemDim_x + tx] = grid[iy * width + (ix - 1 + width) % width];
	}

	if ((tx == blockDim.x - 1) || (ix == width - 1))
	{
		// E
		if (ix == width - 1)
			s_data[(ty + 1) * sharedMemDim_x + tx + 2] = grid[iy * width];
		else
			s_data[(ty + 1) * sharedMemDim_x + tx + 2] = grid[iy * width + (ix + 1) % width];
	}
	
	// Itself
	s_data[(ty + 1) * sharedMemDim_x + tx + 1] = grid[iy * width + ix];

	__syncthreads();

	const int aliveNeighbors = (	
			s_data[ty * sharedMemDim_x + tx	   ] + //NW
			s_data[ty * sharedMemDim_x + tx + 1] + //N
			s_data[ty * sharedMemDim_x + tx + 2] + //NE
			s_data[(ty + 1) * sharedMemDim_x + tx] + //W
			s_data[(ty + 1) * sharedMemDim_x + tx + 2] + //E
			s_data[(ty + 2) * sharedMemDim_x + tx] + //SW
			s_data[(ty + 2) * sharedMemDim_x + tx + 1] + //S
			s_data[(ty + 2) * sharedMemDim_x + tx + 2] //SE   
                        );
	if ((s_data[(ty + 1) * sharedMemDim_x + tx + 1] == 1) && (aliveNeighbors != 2) && (aliveNeighbors != 3))
		grid[iy * width + ix] = 0;
	else if ((s_data[(ty + 1) * sharedMemDim_x + tx + 1] == 0) && (aliveNeighbors == 3))
		grid[iy * width + ix] = 1;
}
