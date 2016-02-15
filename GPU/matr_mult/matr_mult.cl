__kernel void matr_mult(__global const float *x, 
                        __global const float *y, 
                        __global float *restrict z, const int size)
{

	int i, j;

	i = get_global_id(0);// / size;
	j = get_global_id(1);// % size;

	float total=0;

	for (int k=0; k<size; ++k){
		
		total += x[i*size+k]*y[k*size+j];	
	}

	z[i*size+j]=total;
}
