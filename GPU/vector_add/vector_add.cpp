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

CPU is slightly faster than GPU if we do not use mapped buffers.

With mapping, the performance is almost identical
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
const unsigned N = 50000000;
float *input_a=(float *) malloc(sizeof(float)*N);
float *input_b=(float *) malloc(sizeof(float)*N);
float *output=(float *) malloc(sizeof(float)*N);
float *ref_output=(float *) malloc(sizeof(float)*N);
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
int status;


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

     unsigned char **opencl_program=read_file("vector_add.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "vector_add", NULL);
 // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
       N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
        N* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
	cl_event kernel_event,finish_event;
    
    input_a =  (float*) clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, CL_MAP_WRITE,0, N*sizeof(float),0, NULL, &write_event[0], &error_code);
    checkError(error_code, "Failed to map input A");
    input_b = (float*) clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, CL_MAP_WRITE,0, N*sizeof(float),0,NULL, &write_event[1], &error_code);
    checkError(error_code, "Failed to map input B");
    output = (float*) clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ,0, N*sizeof(float),0, NULL,NULL, &error_code);
    checkError(error_code, "Failed to map output");
    
	//Generate random numbers

        for(unsigned j = 0; j < N; ++j) {
              input_a[j] = rand_float();
              input_b[j] = rand_float();
        }
	cout << "Vectors generated." << endl;
	
	cout << "Begin of CPU's computation" << endl;
	//Let the cpu compute the vector sum
        clock_t begin_cpu, end_cpu, begin_gpu, end_gpu;
	struct timespec cpu_begin, cpu_end, gpu_begin, gpu_end;
	double gputime=0, cputime=0;

	clock_gettime( CLOCK_REALTIME, &cpu_begin);
        begin_cpu=clock();
        for(unsigned j = 0; j < N; ++j) {
              ref_output[j] = input_a[j] + input_b[j];
              //printf("ref %f\n",ref_output[j]);
            }
        clock_gettime( CLOCK_REALTIME, &cpu_end);
        end_cpu=clock();
        cputime = (double)( cpu_end.tv_sec - cpu_begin.tv_sec ) + (double)( cpu_end.tv_nsec - cpu_begin.tv_nsec ) / BILLION;
        printf ("CPU took %.2lf seconds to run.\n", cputime );
        printf("CPU took %ld clock cycles to run\n", end_cpu - begin_cpu);
    
    // Set kernel arguments.

    cout << "Preparing GPU for computation" << endl;
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1: 1st input vector");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2: 2nd input vector");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3: output vector");

    const size_t global_work_size = N;
    
    clWaitForEvents(2,write_event);

    clEnqueueUnmapMemObject(queue,input_a_buf,input_a,0,NULL,NULL);
    clEnqueueUnmapMemObject(queue,input_b_buf,input_b,0,NULL,NULL);
    printf("Buffers unmapped\n");

    cout << "Begin of GPU's computation" << endl;
    clock_gettime( CLOCK_REALTIME, &gpu_begin);
    begin_gpu=clock();
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
    &global_work_size, NULL, 0, NULL, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, N* sizeof(float), output, 1, &kernel_event, &finish_event);

   clock_gettime( CLOCK_REALTIME, &gpu_end);
   end_gpu=clock();
   gputime = (double)( gpu_end.tv_sec - gpu_begin.tv_sec ) + (double)( gpu_end.tv_nsec - gpu_begin.tv_nsec ) / BILLION;
   printf ("GPU took %.8lf seconds to run.\n", gputime);
   printf("GPU took %ld clocks to run\n", end_gpu-begin_gpu);
// Verify results.
bool pass = true;

for(unsigned j = 0; j < N && pass; ++j) {
      if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
            j, output[j], ref_output[j]);
        pass = false;
      }
}

printf("GPU is approximately %lf times faster than CPU\n",((float) (end_cpu-begin_cpu)) /((float) (end_gpu - begin_gpu))); 
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
