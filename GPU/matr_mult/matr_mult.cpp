#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
#define BILLION  1000000000L
using namespace std;

/*
Performance notes:

###############################
# With 200x200 square matrices#
###############################

Without using mapped buffers: GPU is approximately 9.34 times faster
With mapped buffers: GPU is approximately 11.84 times faster

###############################
# With 400x400 square matrices#
###############################

With mapped buffers: GPU is approximately 55 times faster

*/


void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
		exit(1);
	}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);
  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)	
		printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;
     cl_int error_code;


//--------------------------------------------------------------------
const unsigned N = 200;
float *input_a=(float *) malloc(sizeof(float)*N*N); //First matrix
float *input_b=(float *) malloc(sizeof(float)*N*N); //Second matrix
float *output=(float *) malloc(sizeof(float)*N*N); //Output matrix
float *ref_output=(float *) malloc(sizeof(float)*N*N); //Reference output matrix
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;
unsigned int i,j;

 //OpenCL initializations 
     clGetPlatformIDs(1, &platform, NULL);
     clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
     clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
     printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

     context_properties[1] = (cl_context_properties)platform;
     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
     context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
     queue = clCreateCommandQueue(context, device, 0, NULL);

     unsigned char **opencl_program=read_file("matr_mult.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "matr_mult", NULL);
    
    // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       N*N *sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N*N *sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N*N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");



    // Map the input and output buffers
    cl_event write_event[2];
    cl_event kernel_event,finish_event;

    input_a =  (float*) clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, CL_MAP_WRITE,0, N*N*sizeof(float),0, NULL, &write_event[0], &error_code);
    checkError(error_code, "Failed to map input A");
    input_b = (float*) clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, CL_MAP_WRITE,0, N*N*sizeof(float),0,NULL, &write_event[1], &error_code);
    checkError(error_code, "Failed to map input B");
    output = (float*) clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ,0, N*N*sizeof(float),0, NULL,NULL, &error_code);
    checkError(error_code, "Failed to map output");

	//Initialize the matrix elements to random values
	printf("Generating random entries for the matrix\n");
	for( i = 0; i < N; ++i) {
		
		for(j = 0; j < N; ++j) {
			
	      		input_a[i*N+j] = rand_float();
	      		input_b[i*N+j] = rand_float();
	      		ref_output[i*N+j]=0;
	      	}	//printf("ref %f\n",ref_output[j]);
	    }
	printf("Matrices both generated\n");


        clock_t cpu_b, cpu_e, gpu_b, gpu_e;
	struct timespec cpu_begin, cpu_end, gpu_begin, gpu_end;
	double gputime=0, cputime=0;

	cout << "Begin of CPU's computation" << endl;
	clock_gettime( CLOCK_REALTIME, &cpu_begin);
	cpu_b=clock();
	for(unsigned i = 0; i < N; ++i) {
		
		for(unsigned j = 0; j < N; ++j) {
			
			for (unsigned k=0; k<N; ++k){
	      			ref_output[i*N+j]+= input_a[i*N+k] * input_b[k*N+j];
			}
	      	}	//printf("ref %f\n",ref_output[j]);
	}
	cpu_e=clock();
	clock_gettime( CLOCK_REALTIME, &cpu_end);
	cout << "CPU calculated the product\n" << endl;
	cputime = (double)( cpu_end.tv_sec - cpu_begin.tv_sec ) + (double)( cpu_end.tv_nsec - cpu_begin.tv_nsec ) / BILLION;
	clock_t clocks_cpu = cpu_e - cpu_b;
  	printf ("CPU took %.2lf seconds to run.\n", cputime);
	printf("CPU took %ld clock cycles to run\n", clocks_cpu);


    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1: 1st input matrix");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2: 2nd input matrix");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3: output matrix");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &N);
    checkError(status, "Failed to set argument 4: size of the square matrices");

    size_t global_work_size[2] = {N,N};
    size_t global_work_offset[2] = {0,0};
    
    clWaitForEvents(2,write_event);

    clEnqueueUnmapMemObject(queue,input_a_buf,input_a,0,NULL,NULL);
    clEnqueueUnmapMemObject(queue,input_b_buf,input_b,0,NULL,NULL);
    clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);
    printf("Buffers unmapped\n");

    cout << "Start of GPU computation" << endl;
    clock_gettime( CLOCK_REALTIME, &gpu_begin);
    gpu_b = clock();
    status = clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset,
         global_work_size, NULL, 1, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, N*N* sizeof(float), output, 1, &kernel_event, &finish_event);
   gpu_e = clock();
   clock_gettime( CLOCK_REALTIME, &gpu_end);
   cout << "End of GPU computation" << endl;

   clock_t gpu_clocks = gpu_e - gpu_b;
   gputime = (double)( gpu_end.tv_sec - gpu_begin.tv_sec ) + (double)( gpu_end.tv_nsec - gpu_begin.tv_nsec ) / BILLION;
   printf ("GPU took %.8lf seconds to run.\n", gputime );
   printf("GPU took %ld clock cycles to run \n", gpu_clocks);

   printf("Conclusion: GPU proved to be approximately %f times faster than CPU\n", clocks_cpu * 1.0 / gpu_clocks);
// Verify results.
bool pass = true;

for (unsigned i = 0; i < N && pass; ++i)
	for(unsigned j = 0; j < N && pass; ++j) {
      		if(fabsf(output[i*N+j] - ref_output[i*N+j]) > 1.0e-5f) {
        		printf("Failed verification @ indexes (%d,%d)\nOutput: %f\nReference: %f\n",i,j, output[i*N+j], ref_output[i*N+j]);
        		pass = false;
      		}
}

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
clReleaseKernel(kernel);
clReleaseCommandQueue(queue);
clReleaseMemObject(input_a_buf);
clReleaseMemObject(input_b_buf);
clReleaseMemObject(output_buf);
clReleaseProgram(program);
clReleaseContext(context);


//--------------------------------------------------------------------






     clFinish(queue);

     return 0;
}
