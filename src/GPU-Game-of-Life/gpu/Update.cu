#include "Update.h"

#include <stdlib.h>

#include "Kernel.h"

void updateGridOnGPU(unsigned char* gameGrid,
                     size_t gameGridWidth, size_t gameGridHeight,
                     size_t blockWidth, size_t blockHeight)
{
    size_t pitch;
	unsigned char* d_grid;
	cudaMallocPitch(&d_grid, &pitch, gameGridWidth * sizeof(unsigned char), gameGridHeight);
	
	cudaMemcpy2D(d_grid, pitch, gameGrid, gameGridWidth * sizeof(unsigned char), gameGridWidth * sizeof(unsigned char), gameGridHeight, cudaMemcpyHostToDevice);

	const int gridWidth = (int)ceil((gameGridWidth) / (float)blockWidth);
	const int gridHeight = (int)ceil((gameGridHeight) / (float)blockHeight);

	const unsigned int sharedMemSize = ((blockWidth + 2) * (blockHeight + 2)) * sizeof(unsigned char); 
 
   	const dim3 grid(gridWidth, gridHeight, 1);
   	const dim3 threads(blockWidth, blockHeight, 1);

    updateGrid<<<grid, threads, sharedMemSize>>>(d_grid, pitch / sizeof(unsigned char), gameGridWidth, gameGridHeight);
	
	cudaMemcpy2D(gameGrid, gameGridWidth * sizeof(int), d_grid, pitch, gameGridWidth * sizeof(unsigned char), gameGridHeight, cudaMemcpyDeviceToHost);
}
