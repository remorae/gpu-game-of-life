#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void playGame(int *gridIn, int intpitch, int width, int height){
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; //row index
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //column index
	
	int tx = threadIdx.x; // For shared memory
	int ty = threadIdx.y;
	extern __shared__ int s_data[];
	int sharedMemDim_x = blockDim.x+2;
	
	/*
	s_data[tx] = 					gridIn[(height-1)*intpitch + width - 1];//NW1 	iy =  0, ix =  0, tx =  0 
	s_data[tx] = 					gridIn[(height-1)*intpitch + ix - 1];	//NW2	iy =  0, ix >  0, tx =  0 
	s_data[tx] = 					gridIn[(iy-1)*intpitch + width - 1];	//NW3 	iy >  0, ix =  0, tx =  0
	s_data[tx] = 					gridIn[(iy-1)*intpitch + ix - 1];	//NW4 	iy >  0, ix >  0, tx =  0 
	s_data[tx + 1] = 				gridIn[(height-1)*intpitch + ix];	//N1  	iy =  0, ix >= 0, tx >= 0 
	s_data[tx + 1] = 				gridIn[(iy-1)*intpitch + ix];		//N2  	iy >  0, ix >= 0, tx >= 0
	s_data[tx + 2] = 				gridIn[(height-1)*intpitch];		//NE1 	iy =  0, ix =  l, tx >= 0 
	s_data[tx + 2] = 				gridIn[(height-1)*intpitch + ix + 1];	//NE2 	iy =  0, ix <  l, tx =  l
	s_data[tx + 2] = 				gridIn[(iy-1)*intpitch];		//NE3 	iy >  0, ix =  l, tx >= 0 
	s_data[tx + 2] = 				gridIn[(iy-1)*intpitch + ix + 1];	//NE4 	iy >  0, ix <  l, tx =  l
	s_data[(ty+1)*sharedMemDim_x + tx] = 		gridIn[iy*intpitch + width - 1];	//W1  	iy >= 0, ix =  0, tx =  0
	s_data[(ty+1)*sharedMemDim_x + tx] = 		gridIn[iy*intpitch + ix - 1];     	//W2  	iy >= 0, ix >  0, tx =  0
	s_data[(ty+1)*sharedMemDim_x + tx + 1] = 	gridIn[iy*intpitch + ix];    		//itself
	s_data[(ty+1)*sharedMemDim_x + tx + 2] = 	gridIn[iy*intpitch];     		//E1  	iy >= 0, ix =  l, tx >= 0
	s_data[(ty+1)*sharedMemDim_x + tx + 2] = 	gridIn[iy*intpitch + ix + 1];     	//E2  	iy >= 0, ix <  l, tx =  l
	s_data[(ty+2)*sharedMemDim_x + tx + 2] =	gridIn[0];				//SE1 	iy =  l, ix =  l, tx >= 0
	s_data[(ty+2)*sharedMemDim_x + tx + 2] = 	gridIn[ix + 1];				//SE2 	iy =  l, ix <  l, tx =  l
	s_data[(ty+2)*sharedMemDim_x + tx + 2] = 	gridIn[(iy+1)*intpitch];		//SE3 	iy <  l, ix =  l, tx >= 0
	s_data[(ty+2)*sharedMemDim_x + tx + 2] = 	gridIn[(iy+1)*intpitch + ix + 1];	//SE4 	iy <  l, ix <  l, tx =  l
	s_data[(ty+2)*sharedMemDim_x + tx + 1] =	gridIn[ix];				//S1  	iy =  l, ix >= 0, tx >= 0
	s_data[(ty+2)*sharedMemDim_x + tx + 1] =	gridIn[(iy + 1)*intpitch + ix];		//S2  	iy <  l, ix >= 0, tx >= 0
	s_data[(ty+2)*sharedMemDim_x + tx] = 		gridIn[width - 1];			//SW1 	iy =  l, ix =  0, tx =  0
	s_data[(ty+2)*sharedMemDim_x + tx] = 		gridIn[ix - 1];				//SW2 	iy =  l, ix >  0, tx =  0 
	s_data[(ty+2)*sharedMemDim_x + tx] = 		gridIn[(iy+1)*intpitch + width - 1];	//SW3 	iy <  l, ix =  0, tx =  0
	s_data[(ty+2)*sharedMemDim_x + tx] = 		gridIn[(iy+1)*intpitch + ix - 1];	//SW4 	iy <  l, ix >  0, tx =  0

	
	2D block logic
	NW: ((ty == 0) && ((tx == 0) || (tx == 1)))
	N:  ((ty == 0) && (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width - 1) && (tx = 0)) || (blockDim.x == 1))
	NE: ((ty == 0) && ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)))

	W:  (tx == 0)

	SW: ((ty == height-1) && ((tx == 0) || (tx == 1)))
	S:  ((ty == height-1) && (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width-1) && (tx == 0)) || (blockDim.x == 1))
	SE: ((ty == height-1) && ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)))

	E:  ((tx == blockDim.x-1) || (ix = width-1))
	*/

	if ((iy < height) && (ix < width)){
		if (ty == 0){
			if (iy == 0){
				if (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width - 1) && (tx == 0)) || (blockDim.x == 1))
					s_data[tx + 1] = gridIn[(height-1)*intpitch + ix];					//N1

				if ((tx==0) || (tx == 1)){
					if (ix == 0)
						s_data[tx] = gridIn[(height-1)*intpitch + width - 1];				//NW1
					else	
						s_data[tx] = gridIn[(height-1)*intpitch + ix - 1];				//NW2
				}

				if ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)){
					if (ix == width-1)
						s_data[tx + 2] = gridIn[(height-1)*intpitch];					//NE1 
					else
						s_data[tx + 2] = gridIn[(height-1)*intpitch + ix + 1];				//NE2
				}
			}
			else{
				if (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width - 1) && (tx == 0)) || (blockDim.x == 1))
					s_data[tx + 1] = gridIn[(iy-1)*intpitch + ix];						//N2

				if ((tx==0) || (tx == 1)){
					if (ix == 0)
						s_data[tx] = gridIn[(iy-1)*intpitch + width - 1];				//NW3
					else
						s_data[tx] = gridIn[(iy-1)*intpitch + ix - 1];					//NW4
				}

				if ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)){
					if (ix == width-1)
						s_data[tx + 2] = gridIn[(iy-1)*intpitch];					//NE3 
					else
						s_data[tx + 2] = gridIn[(iy-1)*intpitch + ix + 1];				//NE4
				}
			}
		}

		if ((ty == blockDim.y-1) || (iy == height-1)){
			if (iy == height-1){
				if (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width-1) && (tx == 0)) || (blockDim.x == 1))
					s_data[(ty+2)*sharedMemDim_x + tx + 1] = gridIn[ix];					//S1
	
				if ((tx == 0) || (tx == 1)){
					if (ix == 0)
						s_data[(ty+2)*sharedMemDim_x + tx] = gridIn[width - 1];				//SW1
					else
						s_data[(ty+2)*sharedMemDim_x + tx] = gridIn[ix - 1];				//SW2
				}

				if ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)){
					if (ix == width-1)
						s_data[(ty+2)*sharedMemDim_x + tx + 2] = gridIn[0];				//SE1 
					else
						s_data[(ty+2)*sharedMemDim_x + tx + 2] = gridIn[ix + 1];			//SE2
				}
			}
			else{
				if (((tx != 0) && (tx != blockDim.x-1) && (ix != width-1)) || ((ix == width-1) && (tx == 0)) || (blockDim.x == 1))
					s_data[(ty+2)*sharedMemDim_x + tx + 1] = gridIn[(iy + 1)*intpitch + ix];		//S2

				if ((tx == 0) || (tx == 1)){
					if (ix == 0)
						s_data[(ty+2)*sharedMemDim_x + tx] = gridIn[(iy+1)*intpitch + width - 1];	//SW3
					else
						s_data[(ty+2)*sharedMemDim_x + tx] = gridIn[(iy+1)*intpitch + ix - 1];		//SW4
				}

				if ((tx == blockDim.x-1) ||  (tx == blockDim.x-2) || (ix == width-1) || (ix == width-2)){
					if (ix == width-1)
						s_data[(ty+2)*sharedMemDim_x + tx + 2] = gridIn[(iy+1)*intpitch];		//SE3 
					else
						s_data[(ty+2)*sharedMemDim_x + tx + 2] = gridIn[(iy+1)*intpitch + ix + 1];	//SE4
				}
			}
		}

		if (tx == 0){
			if (ix == 0)
				s_data[(ty+1)*sharedMemDim_x] = gridIn[iy*intpitch + width - 1];				//W1
			else
				s_data[(ty+1)*sharedMemDim_x] = gridIn[iy*intpitch + ix - 1];  			   		//W2
		}

		if ((tx == blockDim.x-1) || (ix == width-1)){
			if (ix == width-1)
				s_data[(ty+1)*sharedMemDim_x + tx + 2] = gridIn[iy*intpitch];					//E1
			else
				s_data[(ty+1)*sharedMemDim_x + tx + 2] = gridIn[iy*intpitch + ix + 1];     			//E2
		}
	
		s_data[(ty+1)*sharedMemDim_x + tx + 1] = gridIn[iy*intpitch + ix];    						//itself
	}
	__syncthreads();

	/*if ((blockIdx.y == 0) && (blockIdx.x == 0) && (threadIdx.x == 0) && (threadIdx.y == 0)){
		int sharedMemDim_y = blockDim.y+2;		
		printf("block_y: %d, block_x: %d\n", 0, 0);
		for(int i=0; i<sharedMemDim_y; i++){
			for(int j=0; j<sharedMemDim_x; j++){
				printf("%d ", s_data[i*sharedMemDim_x + j]);
			}
			
			printf("\n");
		}

		printf("\n");
	}*/

	if ((ix<width) && (iy<height)){
		int sum = (	
				s_data[ty*sharedMemDim_x + tx		] +	//NW
				s_data[ty*sharedMemDim_x + tx + 1	] +	//N
				s_data[ty*sharedMemDim_x + tx + 2	] +	//NE
				s_data[(ty+1)*sharedMemDim_x + tx	] +	//W
				s_data[(ty+1)*sharedMemDim_x + tx + 2	] +	//E
				s_data[(ty+2)*sharedMemDim_x + tx	] +	//SW
				s_data[(ty+2)*sharedMemDim_x + tx + 1	] +	//S
				s_data[(ty+2)*sharedMemDim_x + tx + 2	]	//SE   
                           );
		if ((s_data[(ty+1)*sharedMemDim_x + tx + 1] == 1) && (sum != 2) && (sum != 3))
			gridIn[iy*intpitch + ix] = 0;
		else if ((s_data[(ty+1)*sharedMemDim_x + tx + 1] == 0) && (sum == 3))
			gridIn[iy*intpitch + ix] = 1;
	}
}
