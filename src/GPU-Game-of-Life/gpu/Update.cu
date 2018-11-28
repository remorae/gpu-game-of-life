#include "Update.h"

#include <stdio.h>
#include <stdlib.h>

#include "Kernel.h"

void updateGridOnGPU(unsigned char* gameGrid,
                     const size_t gameGridWidth, const size_t gameGridHeight,
                     const size_t blockWidth, const size_t blockHeight)
{
	// Our only guarantee up to this point is that the total number of threads per block is <= 1024.
	// We need to ensure large grid sizes can be updated even if requiring multiple kernel calls.

	// Step 1: Determine how many iterations it's going to take, along with an adjusted # of blocks per kernel call

	cudaError_t error;
	size_t maxNumBlocksX = (size_t)ceil((gameGridWidth) / (float)blockWidth);
	size_t maxNumBlocksY = (size_t)ceil((gameGridHeight) / (float)blockHeight);
	size_t iterationsNeeded = 1;
	size_t blocksPerIteration = maxNumBlocksX * maxNumBlocksY;

	while (blocksPerIteration > 65536)
	{
		++iterationsNeeded;
		blocksPerIteration /= 2;
		if (maxNumBlocksY == 1)
			maxNumBlocksX /= 2;
		else
			maxNumBlocksY /= 2;
	}

	//printf("Game grid: %lux%lu Block size: %lux%lu maxBlocksPerKernel: %lux%lu\n", gameGridWidth, gameGridHeight, blockWidth, blockHeight, maxNumBlocksX, maxNumBlocksY);
	//printf("Iters: %lu blocksPerIter: %lu\n", iterationsNeeded, blocksPerIteration);
	
	// Step 2: Setup device memory and the necessary shared memory size as usual

	unsigned char* d_destGrid, d_srcGrid;
	cudaMalloc(&d_srcGrid, gameGridWidth * gameGridHeight * sizeof(unsigned char));
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("cudaMalloc error: %s\n", cudaGetErrorString(error));
	cudaMalloc(&d_destGrid, gameGridWidth * gameGridHeight * sizeof(unsigned char));
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("cudaMalloc error: %s\n", cudaGetErrorString(error));
	
	cudaMemcpy(d_srcGrid, gameGrid, gameGridWidth * gameGridHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("cudaMemcpy h2d error: %s\n", cudaGetErrorString(error));
	
	cudaMemcpy(d_destGrid, gameGrid, gameGridWidth * gameGridHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("cudaMemcpy h2d error: %s\n", cudaGetErrorString(error));

	const unsigned int sharedMemSize = ((blockWidth + 2) * (blockHeight + 2)) * sizeof(unsigned char);

	//printf("Shared mem: %lu\n", (long unsigned)sharedMemSize);

	// Step 3: Determine how many blocks are needed total in the X and Y directions.
	// This can greatly reduce the number of blocks created when updating areas along the right and bottom sides of the grid.
	// Essentially, the final call(s) in the X and Y directions have smaller CUDA grid size parameters.
 
	const size_t blocksNeededX = (size_t)ceil(gameGridWidth / (float)blockWidth);
	const size_t blocksNeededY = (size_t)ceil(gameGridHeight / (float)blockHeight);
	const size_t iterationsNeededX = (size_t)ceil(blocksNeededX / (float)maxNumBlocksX);

	// Step 4: Run the necessary number of kernel calls to update the grid

   	const dim3 cudaBlockDimensions(blockWidth, blockHeight, 1);

	for (size_t i = 0; i < iterationsNeeded; ++i)
	{
		// Iterations move along the game grid as if it were linear, so figure out where the current "tile" (kernel tile, not block tile) is.

		const size_t gridX = i % iterationsNeededX;
		const size_t gridY = i / iterationsNeededX;

		// Reduce the grid parameters if possible (see step 3).

		const size_t cudaGridWidth = min(blocksNeededX - (gridX * maxNumBlocksX), maxNumBlocksX);
		const size_t cudaGridHeight = min(blocksNeededY - (gridY * maxNumBlocksY), maxNumBlocksY);

		// Call the kernel

		//printf("Iter %lu: @(%lu,%lu) blocks: %lux%lu\n", i, gridX, gridY, cudaGridWidth, cudaGridHeight);

   		const dim3 cudaGridDimensions(cudaGridWidth, cudaGridHeight, 1);

		updateGrid<<<cudaGridDimensions, cudaBlockDimensions, sharedMemSize>>>(d_destGrid, d_srcGrid
															gameGridWidth,
															gameGridHeight,
															gridX * maxNumBlocksX * blockWidth, // Offset for iteration
															gridY * maxNumBlocksY * blockHeight // Offset for iteration
															);
		error = cudaGetLastError();
		if (error != cudaSuccess)
			printf("kernel error: %s\n", cudaGetErrorString(error));
	}

	// Step 5: Copy results back to host and return them
	
	cudaMemcpy(gameGrid, d_destGrid, gameGridWidth * gameGridHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		printf("cudaMemcpy d2h error: %s\n", cudaGetErrorString(error));
	cudaFree(d_destGrid);
	cudaFree(d_srcGrid);
}
