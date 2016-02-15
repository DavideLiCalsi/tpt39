#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
#define BILLION  1000000000L

using namespace cv;
using namespace std;

//Begin utility functions

void copyArrays(int* v1, int* v2, int len){

	for (int i=0; i<len; ++i){
		v1[i]=v2[i];
	}
}

int* frame2intmatrix(Mat m){
	
	int* result = (int*) malloc(sizeof(int) * m.cols*m.rows);
	
	for (int i =0; i < m.rows; ++i){
		
		for (int j=0; j <m.cols; ++j){
			result[i*m.cols + j] = m.at<int>(i,j);		
		}	
	}

	return result;
	
}

Mat arrayto2Dmat(int* array, int rows, int columns){
	
	Mat m = Mat(rows, columns, CV_32S);

	for (int i=0; i<rows; ++i){
		for (int j=0; j<columns; ++j){
			m.at<int>(i,j) = array[i*columns + j];
		}
	}

	return m;
}

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
		printf("Error code: %d\n%s\n",status,msg);
}
//end utility functions

#define SHOW
int main(int, char**)
{
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME_GPU = "./output_gpu.avi";   // Form the new name with container
    const string NAME_CPU = "output_cpu.avi";
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);
	cout << "SIZE:" << S << endl;
	
    VideoWriter outputVideoGPU, outputVideoCPU;                                        // Open the output for both CPU and GPU
        outputVideoGPU.open(NAME_GPU, ex, 25, S, true);
	outputVideoCPU.open(NAME_CPU, ex, 25, S, true);

    if (!outputVideoCPU.isOpened() || !outputVideoGPU.isOpened() )
    {
        cout  << "Could not open the output video for write"<< endl;
        return -1;
    }
	time_t start,end;
	double diff,totGPU=0, totCPU=0;
	int count=0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif
	
     //OpenCL initialization
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
     int status;
     cl_int errcode;

     cl_event write_event[2];
     cl_event kernel_event;
     int gauss_filter = 0, sobel_filter=1;

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

     unsigned char **opencl_program=read_file("filter.cl");
     program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
     if (program == NULL)
	{
         printf("Program creation failed\n");
         return 1;
	}	
     int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	 if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
     kernel = clCreateKernel(program, "convolution", &status);
	if (kernel==NULL){
		printf("Failed to create kernel, error code %d\n", status);
	}

     //Create input buffers
     cl_mem frame2filter_buf, output_buf;

     frame2filter_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, S.height*S.width *sizeof(int), NULL, &status);
     checkError(status, "Failed to create buffer for input frame");

     output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, S.height*S.width *sizeof(int), NULL, &status);
     checkError(status, "Failed to create buffer for output frame");


     unsigned argi = 0;
	
    	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &frame2filter_buf);
    	checkError(status, "Failed to set argument 1: frame buffer");

    	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    	checkError(status, "Failed to set argument 2: output buffer");

    	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &(S.width) );
    	checkError(status, "Failed to set argument 3: frame width");

    	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &(S.height) );
    	checkError(status, "Failed to set argument 4: frame height");

	struct timespec cpu_begin, cpu_end, gpu_begin, gpu_end;
	double gputime=0, cputime=0;

    //While loop that processes the frames. We'll process each frame first with the GPU, then with the CPU, in this loop
    while (true) {

        Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 299) break;
        camera >> cameraFrame;
        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe,edge_x,edge_y,edge,edge_inv;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

	
	//Start gpu operations
	cout << "GPU processing frame " << count << endl;
	//int_matr is a copy of grayframe but converted to integer values
	Mat int_matr;
	grayframe.convertTo(int_matr,CV_32S); 
	int* source_image = frame2intmatrix(int_matr); 

    	
	/*
	##################################
	# Code for the 1st gaussian blur #
	##################################
	*/

	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &(gauss_filter) );
    	checkError(status, "Failed to set argument 5: filter type");

	int* frame2filter = (int *)clEnqueueMapBuffer(queue, frame2filter_buf, CL_TRUE,CL_MAP_WRITE,0, S.height*S.width *sizeof(int), 0, NULL, &write_event[0],&errcode);
     	checkError(errcode, "Failed to map input");

     	int* output = (int *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,CL_MAP_READ,0, S.height*S.width *sizeof(int), 0, NULL, NULL,&errcode);
     	checkError(errcode, "Failed to map output");

	copyArrays(frame2filter, source_image,S.width*S.height);

	clEnqueueUnmapMemObject(queue,frame2filter_buf,frame2filter,0,NULL,NULL);
	clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);

	size_t global_work_size[2] = {(unsigned) grayframe.rows, (unsigned) grayframe.cols};
	
	time(&start);
	clock_gettime( CLOCK_REALTIME, &gpu_begin);   
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 1, write_event, &kernel_event);
    	checkError(status, "Failed to launch kernel");

	/*
	##################################
	# Code for the 2nd gaussian blur #
	##################################
	*/
	
	frame2filter = (int *)clEnqueueMapBuffer(queue, frame2filter_buf, CL_TRUE,CL_MAP_WRITE,0, S.height*S.width *sizeof(int), 0, NULL, &write_event[0],&errcode);
     	checkError(errcode, "Failed to map input");

     	output = (int *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,CL_MAP_READ,0, S.height*S.width *sizeof(int), 0, NULL, NULL,&errcode);
     	checkError(errcode, "Failed to map output");

	copyArrays(frame2filter, output,S.width*S.height);

	clEnqueueUnmapMemObject(queue,frame2filter_buf,frame2filter,0,NULL,NULL);
	clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);
	
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 1, write_event, &kernel_event);
    	checkError(status, "Failed to launch kernel");

	/*
	##################################
	# Code for the 3rd gaussian blur #
	##################################
	*/


	frame2filter = (int *)clEnqueueMapBuffer(queue, frame2filter_buf, CL_TRUE,CL_MAP_WRITE,0, S.height*S.width *sizeof(int), 0, NULL, &write_event[0],&errcode);
     	checkError(errcode, "Failed to map input");

     	output = (int *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,CL_MAP_READ,0, S.height*S.width *sizeof(int), 0, NULL, NULL,&errcode);
     	checkError(errcode, "Failed to map output");

	copyArrays(frame2filter, output,S.width*S.height);

	clEnqueueUnmapMemObject(queue,frame2filter_buf,frame2filter,0,NULL,NULL);
	clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);
	
	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 1, write_event, &kernel_event);
    	checkError(status, "Failed to launch kernel");
	
	/*
	##################################
	#    Code for the Sobel filter   #
	##################################
	*/

	
	Mat after3blurs = arrayto2Dmat(output, S.height, S.width);
	frame2filter = (int *)clEnqueueMapBuffer(queue, frame2filter_buf, CL_TRUE,CL_MAP_WRITE,0, S.height*S.width *sizeof(int), 0, NULL, &write_event[0],&errcode);
     	checkError(errcode, "Failed to map input");

     	output = (int *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,CL_MAP_READ,0, S.height*S.width *sizeof(int), 0, NULL, NULL,&errcode);
     	checkError(errcode, "Failed to map output");

	copyArrays(frame2filter, output,S.width*S.height);

	clEnqueueUnmapMemObject(queue,frame2filter_buf,frame2filter,0,NULL,NULL);
	clEnqueueUnmapMemObject(queue,output_buf,output,0,NULL,NULL);

	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &(sobel_filter) );
    	checkError(status, "Failed to set argument 5: filter type");

	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 1, write_event, &kernel_event);
    	checkError(status, "Failed to launch kernel");
	
	output = (int *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE,CL_MAP_READ,0, S.height*S.width *sizeof(int), 0, NULL, NULL,&errcode);
     	checkError(errcode, "Failed to map input");
	clock_gettime( CLOCK_REALTIME, &gpu_end);
	time(&end);

	cout << "GPU done processing frame " << count << endl;
    	//end gpu operations
	
	//Final processing for GPU's results and store the computed frame
	edge = arrayto2Dmat(output, grayframe.rows, grayframe.cols);
	edge.convertTo(edge, CV_8U);
	threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
	after3blurs.convertTo(after3blurs, CV_8U);
	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
	after3blurs.copyTo(displayframe, edge),
        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
	outputVideoGPU << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
	diff = difftime (end,start);
	totGPU+=diff;
	gputime += (double)( gpu_end.tv_sec - gpu_begin.tv_sec ) + (double)( gpu_end.tv_nsec - gpu_begin.tv_nsec ) / BILLION;


	//CPU processing start
	cout << "CPU processing frame " << count << endl;
	time(&start);
	clock_gettime( CLOCK_REALTIME, &cpu_begin);
	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
	Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
	Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
	addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
	clock_gettime( CLOCK_REALTIME, &cpu_end);	
	time (&end);
        cvtColor(edge, edge_inv, CV_GRAY2BGR);
    	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
	grayframe.copyTo(displayframe,edge);
        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
	outputVideoCPU << displayframe;

	diff = difftime (end,start);
	totCPU+=diff;
	cputime += (double)( cpu_end.tv_sec - cpu_begin.tv_sec ) + (double)( cpu_end.tv_nsec - cpu_begin.tv_nsec ) / BILLION;
	cout << "CPU done processing frame " << count << endl;

	}

	outputVideoGPU.release();
	outputVideoCPU.release();
	camera.release();
  	printf ("GPU's FPS %.2lf .\n", 299.0/gputime );
	printf ("CPU's FPS %.2lf .\n", 299.0/cputime );
	printf("GPU took %lf seconds\n", gputime);
	printf("CPU took %lf seconds\n", cputime);

	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(frame2filter_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);

    return EXIT_SUCCESS;

}
