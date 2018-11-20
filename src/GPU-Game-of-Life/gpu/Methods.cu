#include "Methods.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void initializeArray(int* arr, int width, int height)
{
	size_t i, j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{  
			arr[i * width + j] = 0;
		}
	}
}

void printArray(int* arr, int rows, int cols, int shouldPrint){
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


