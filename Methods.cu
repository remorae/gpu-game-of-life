#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Initialize the two arrays referenced by the first two parameters in preparation for 
 * jacobi iteration. The width and height of the arrays are given by the integer parameters.
 * Border elements are set to 5.0 for both arrays, and the interior elements of a1 are
 * set to 1.0.  Interior elements of a2 are not initialized.
 */
void initializeArrays(int *a1, int width, int height){

   int i, j;

   for(i=0; i<height; i++){
      for(j=0; j<width; j++){
      
        if(i==0 || j ==0 || i==height-1 || j==width-1){ 
         
            a1[i*width + j] = 1;
         }
	else {
         
            a1[i*width + j] = 0;
         }
      }
   }
}

/* Print the 2D array of floats referenced by the first parameter. The second and third
 * parameters specify its dimensions, while the last argument indicates whether printing
 * is actually descired at all. No output is produced if shouldPrint == 0.
 */
void printArray(int *arr, int rows, int cols, int shouldPrint){
   if (!shouldPrint)
      return;
          
   int i,j;

   for(i=0; i<rows; i++){
      for(j=0; j<cols; j++){
      
         printf("%d ", arr[i*cols + j]);
      }
      printf("\n");
   }

   printf("\n");
}

int strToInt(char * str){
	int length = strlen(str);
	int integer = 0;
	for (int i=0; i<length; i++){
		integer = integer + (str[i]-'0')*pow((double)10, (double)(length-1-i));
	}
	return integer;
}


