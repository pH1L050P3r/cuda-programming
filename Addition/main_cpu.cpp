#include<iostream>
#include <cstdlib>
#include <assert.h>

void init_arr(int* array, unsigned int size){
	for(int i = 0; i < size; i++)
		array[i] = rand() % 100;	
}

void check_array_sum(int* arr_first, int* arr_second, int* arr_third, unsigned int size){
	for(int i = 0; i < size; i++)
		assert(arr_first[i] + arr_second[i] == arr_third[i]);
}

void sum(int* arr_first, int* arr_second, int* arr_third, unsigned int size){
	for(int i = 0; i < size; i++)
		arr_third[i] = arr_first[i] + arr_second[i];
}

int main(){
	int size = 1024;
	int* first = (int *)malloc(size * sizeof(int));
	int* second = (int *)malloc(size * sizeof(int));
	int* third = (int *)malloc(size * sizeof(int));

	init_arr(first, size);
	init_arr(second, size);
	
	sum(first, second, third, size);
	check_array_sum(first, second, third, size);	

	printf("SUCCESS\n");
	return 0;
}
