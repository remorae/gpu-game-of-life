#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>
#include <helper_timer.h>
#include "GameOfLife.h"
#include "Methods.h"

void usage(){

	fprintf(stderr, "user defined w/ print: ./life gameGridWidth gameGridHeight blockWidth blockHeight numOfPasses p\n");
	fprintf(stderr, "user defined:  	./life gameGridWidth gameGridHeight blockWidth blockHeight numOfPasses\n");
	fprintf(stderr, "auto: 			./life gameGridWidth gameGridHeight numOfPasses\n");
	fprintf(stderr, "auto w/ print:		./life gameGridWidth gameGridHeight numOfPasses p\n");
	fprintf(stderr, "default: 		./life d\n");
	fprintf(stderr, "default w/ print: 	./life d p\n");
	exit(1);
}

int main (int argc, const char * argv[]) {
	// Initialize certain parameters for the game of life.
	int gameGridHeight, gameGridWidth, blockHeight, blockWidth, numOfPasses, print;
	
	if ((argc == 2) && (strcmp(argv[1], "d") == 0)){
		gameGridWidth = 32;
		gameGridHeight = 10;
		blockWidth = 32;
		blockHeight = 10;
		numOfPasses = 18;
		print = 0;
	}
	else if ((argc == 3) && (strcmp(argv[1], "d") == 0) && (strcmp(argv[2], "p") == 0)){
		gameGridWidth = 32;
		gameGridHeight = 10;
		blockWidth = 32;
		blockHeight = 10;
		numOfPasses = 18;
		print = 1;
	}
	else if (argc == 4){
		gameGridWidth = strToInt((char *)argv[1]);
		gameGridHeight = strToInt((char *)argv[2]);
		numOfPasses = strToInt((char *)argv[3]);
		print = 0;
		if ((gameGridWidth < 1) || (gameGridHeight < 1) || (numOfPasses < 1))
			usage();
		if (gameGridWidth*gameGridHeight < 1024){
			blockWidth = gameGridWidth;
			blockHeight = gameGridHeight;
		}
		else{
			if (gameGridWidth < 32){
				blockWidth = gameGridWidth;
				blockHeight = 1024/blockWidth;
			}
			else if (gameGridHeight < 32){
				blockHeight = gameGridHeight;
				blockWidth = 1024/blockHeight;
			}
			else{
				blockHeight = 32;
				blockWidth = 32;
			}
		}		
	}
	else if (argc == 5){
		gameGridWidth = strToInt((char *)argv[1]);
		gameGridHeight = strToInt((char *)argv[2]);
		numOfPasses = strToInt((char *)argv[3]);
		print = 1;
		if ((gameGridWidth < 1) || (gameGridHeight < 1) || (numOfPasses < 1) || (strcmp(argv[4], "p") != 0))
			usage();
		if (gameGridWidth*gameGridHeight < 1024){
			blockWidth = gameGridWidth;
			blockHeight = gameGridHeight;
		}
		else{
			if (gameGridWidth < 32){
				blockWidth = gameGridWidth;
				blockHeight = 1024/blockWidth;
			}
			else if (gameGridHeight < 32){
				blockHeight = gameGridHeight;
				blockWidth = 1024/blockHeight;
			}
			else{
				blockHeight = 32;
				blockWidth = 32;
			}
		}		
	}
	else if (argc == 6){
		gameGridWidth = strToInt((char *)argv[1]);
		gameGridHeight = strToInt((char *)argv[2]);
		blockWidth = strToInt((char *)argv[3]);
		blockHeight = strToInt((char *)argv[4]);
		numOfPasses = strToInt((char *)argv[5]);
		if ((gameGridWidth < 1) || (gameGridHeight < 1) || (numOfPasses < 1) || (blockWidth < 1) || (blockHeight < 1))
			usage();
	}
	else if (argc == 7){
		gameGridWidth = strToInt((char *)argv[1]);
		gameGridHeight = strToInt((char *)argv[2]);
		blockWidth = strToInt((char *)argv[3]);
		blockHeight = strToInt((char *)argv[4]);
		numOfPasses = strToInt((char *)argv[5]);
		print = 1;
		if ((gameGridWidth < 1) || (gameGridHeight < 1) || (numOfPasses < 1) || (blockWidth < 1) || (blockHeight < 1) || (strcmp(argv[6], "p") != 0))
			usage();
	}
	else
		usage();

	// Create a grid (2d array of values) for Game of Life
	int *gameGridIn = (int *) malloc(gameGridHeight * gameGridWidth * sizeof(int));
	// Pause (1) or play (1)
	bool pause = 0; 
	// Initialize the grid
	initializeArrays(gameGridIn, gameGridWidth, gameGridHeight);
	
	size_t pitch;

	// allocate device memory for data in
	int *d_gameGridIn;
	cudaMallocPitch( (void**) &d_gameGridIn, &pitch, gameGridWidth * sizeof(int), gameGridHeight);
	
	// copy host memory to device memory for data in
	cudaMemcpy2D( d_gameGridIn, pitch, gameGridIn, gameGridWidth * sizeof(int), gameGridWidth * sizeof(int), gameGridHeight, cudaMemcpyHostToDevice);

	int gridWidth = (int) ceil( (gameGridWidth) / (float)blockWidth);
	int gridHeight = (int) ceil( (gameGridHeight) / (float)blockHeight);
	printf("block width: %d, block height: %d, grid width: %d, grid height: %d,\n\n", blockWidth, blockHeight, gridWidth, gridHeight);

	// Each block gets a shared memory region of this size.
	unsigned int shared_mem_size = ((blockWidth + 2) * (blockHeight+2)) * sizeof(int); 

	// Format the grid, which is a collection of blocks. 
   	dim3  grid( gridWidth, gridHeight, 1);
   
   	// Format the blocks. 
   	dim3  threads( blockWidth, blockHeight, 1);

	// When game is paused - allow the user to modify the grid values
	// When game is played - make the grid follow the rules
	printf("Starting grid:\n");	
	printArray(gameGridIn, gameGridHeight, gameGridWidth, 1);
	for (int i = 0; (i < numOfPasses); i++){
		//Pausing logic
		/*if (i+1 == 3)
			pause = 1;*/
		//execute the kernel
		if (!pause)
			playGame<<< grid, threads, shared_mem_size >>>( d_gameGridIn, pitch/sizeof(int), gameGridWidth, gameGridHeight);
		else
			i--;
		//Print the array
		if (print == 1){
			cudaMemcpy2D( gameGridIn, gameGridWidth * sizeof(int), d_gameGridIn, pitch, gameGridWidth * sizeof(int), gameGridHeight, cudaMemcpyDeviceToHost);
			printf("Grid Generation: %d\n", i+1);
			printArray(gameGridIn, gameGridHeight, gameGridWidth, 1);
		}
	}

	cudaThreadSynchronize();
	
	cudaMemcpy2D( gameGridIn, gameGridWidth * sizeof(int), d_gameGridIn, pitch, gameGridWidth * sizeof(int), gameGridHeight,cudaMemcpyDeviceToHost);
	if (print!=1){	
		printf("Grid Generation: %d\n", numOfPasses);
		printArray(gameGridIn, gameGridHeight, gameGridWidth, 1);
	}
}
