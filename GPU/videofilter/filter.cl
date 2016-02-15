#define GAUSS_FILTER 0
#define SOBEL_FILTER 1
__kernel void convolution(__global int* a, 
                        __global int* z, const int img_width, const int img_height, const int filter_type)
{


	int i = get_global_id(0);
	int j = get_global_id(1);
	int result;
	int gfilter[9] = {1,2,1,2,4,2,1,2,1};	
	int sob_h[9] = {-3,0,3,-10,0,10, -3, 0, 3};
	int sob_v[9] = {-3,-10,-3,0,0,0, 3, 10, 3};
	
	result = 0;
	
	if ( filter_type == 0 ){
		result += a[ (i-1)*img_width + (j-1) ] * gfilter[0];
		result += a[ (i-1)*img_width + (j) ] * gfilter[1];
		result += a[ (i-1)*img_width + (j+1) ] * gfilter[2];
		result += a[ (i)*img_width + (j-1) ] * gfilter[3];
		result += a[ (i)*img_width + (j) ] * gfilter[4];
		result += a[ (i)*img_width + (j+1) ] * gfilter[5];
		result += a[ (i+1)*img_width + (j-1) ] * gfilter[6];
		result += a[ (i+1)*img_width + (j) ] * gfilter[7];
		result += a[ (i+1)*img_width + (j+1) ] * gfilter[8];
		z[i*img_width + j] = result / 16;
	}
	if ( filter_type == 1 ){
		result += a[ (i-1)*img_width + (j-1) ] * sob_h[0];
		result += a[ (i-1)*img_width + (j) ] * sob_h[1];
		result += a[ (i-1)*img_width + (j+1) ] * sob_h[2];
		result += a[ (i)*img_width + (j-1) ] * sob_h[3];
		result += a[ (i)*img_width + (j) ] * sob_h[4];
		result += a[ (i)*img_width + (j+1) ] * sob_h[5];
		result += a[ (i+1)*img_width + (j-1) ] * sob_h[6];
		result += a[ (i+1)*img_width + (j) ] * sob_h[7];
		result += a[ (i+1)*img_width + (j+1) ] * sob_h[8];

		int h_edge = result;
		
		result=0;

		result += a[ (i-1)*img_width + (j-1) ] * sob_v[0];
		result += a[ (i-1)*img_width + (j) ] * sob_v[1];
		result += a[ (i-1)*img_width + (j+1) ] * sob_v[2];
		result += a[ (i)*img_width + (j-1) ] * sob_v[3];
		result += a[ (i)*img_width + (j) ] * sob_v[4];
		result += a[ (i)*img_width + (j+1) ] * sob_v[5];
		result += a[ (i+1)*img_width + (j-1) ] * sob_v[6];
		result += a[ (i+1)*img_width + (j) ] * sob_v[7];
		result += a[ (i+1)*img_width + (j+1) ] * sob_v[8];

		int v_edge = result;
	
		z[i*img_width + j] = (h_edge + v_edge) / 2;
	}
	

}
