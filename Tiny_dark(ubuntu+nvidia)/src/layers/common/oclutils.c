#include<oclutils.h>
#include<math.h>
#include<string.h>
struct timeval start;
struct timeval end;
const size_t local_work_size1d[1] = {BLOCK_SIZE * BLOCK_SIZE};
const size_t local_work_size2d[2] = {BLOCK_SIZE, BLOCK_SIZE};
const size_t local_work_size3d[3] = {BLOCK_SIZE, BLOCK_SIZE , 1};
cl_context context;
cl_device_id device;
cl_command_queue commandQueue;
cl_program Program;
int ocl_index = 0;
void time_start(void)
{
	gettimeofday(&start, NULL);
} 

double time_elapse(void)
{
	 double elapsed;
	 
	 gettimeofday(&end, NULL);
	 elapsed = ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec) / 1000.0;
	 return elapsed;
} 
  double get_kernel_runtime(cl_event kernel_event)
  {
	  cl_ulong start_time = (cl_ulong)0;
	  cl_ulong end_time = (cl_ulong)0;
	  double run_time;
	  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	  run_time = (double)(end_time - start_time) * 1.0e-6;
	  return run_time;
  }
 char* read_kernel(const char* filename, size_t* kernel_length)
 {
	FILE* fp = fopen(filename, "rb") ;
	if(NULL == fp)
		return NULL;
	 fseek(fp, 0, SEEK_END);
	 size_t file_length = ftell(fp) + 1;
	 fseek(fp, 0, SEEK_SET);
	 char* kernel = (char*)malloc(file_length * sizeof(char));
	 if(!fread(kernel, sizeof(char), file_length - 1, fp))
	 {
		 fclose(fp);
		 free(kernel);
		 return NULL;
	 }
	 if(NULL !=kernel_length)
		 *kernel_length = file_length;
	 fclose(fp);
	kernel[file_length - 1] = '\0';
	return kernel;
 }
 bool check_errNum(cl_int errNum)
 {
 	if(errNum == CL_SUCCESS)
 		return 1;
 	else
 	{
 			switch (errNum)
 		{
 			case CL_DEVICE_NOT_FOUND:
 				printf("errNum : %d, 未发现与条件匹配的OpenCL设备\n", errNum);
 				break;
 			case CL_DEVICE_NOT_AVAILABLE:
 				printf("errNum : %d, OpenCL设备目前不可用\n", errNum);
 				break;
 			case CL_COMPILER_NOT_AVAILABLE:
 				printf("errNum : %d, OpenCL编译器不可用\n", errNum);
 				break;
 			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
 				printf("errNum : %d, 无法为内存对象分配空间\n", errNum);
 				break;
 			case CL_OUT_OF_RESOURCES:
 				printf("errNum : %d, 设备上没有足够的资源\n", errNum);
 				break;
 			case CL_OUT_OF_HOST_MEMORY:
 				printf("errNum : %d, 主机上没有足够的内存\n", errNum);
 				break;
 			case CL_PROFILING_INFO_NOT_AVAILABLE:
 				printf("errNum : %d, 无法得到时间的性能评测信息或者命令队列不支持性能评测\n", errNum);
 				break;
 			case CL_MEM_COPY_OVERLAP:
 				printf("errNum : %d, 两个缓冲区在同一内存区域重叠\n", errNum);
 				break;
 			case CL_IMAGE_FORMAT_MISMATCH:
 				printf("errNum : %d, 图像未采用相同的格式(图像格式不匹配)\n", errNum);
 				break;
 			case CL_IMAGE_FORMAT_NOT_SUPPORTED:
 				printf("errNum : %d, 不支持指定的图像格式\n", errNum);
 				break;
 			case CL_BUILD_PROGRAM_FAILURE:
 				printf("errNum : %d, 无法为程序构建可执行代码\n", errNum);
 				break;
 			case CL_MAP_FAILURE:
 				printf("errNum : %d, 内存区域无法映射到主机内存\n", errNum);
 				break;
 			case CL_MISALIGNED_SUB_BUFFER_OFFSET:
 				printf("errNum : %d, 上下文中没有设备关联的缓冲初始值为CL_DEVICE_MEM_BASE_ADDR_ALIGN\n", errNum);
 				break;
 			case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
 				printf("errNum : %d, 由clWaitForEvent()返回,时间列表中任意事件的执行状态为一个复数\n", errNum);
 				break;
 			case CL_INVALID_VALUE:
 				printf("errNum : %d, 命令的一个或多个参数值不合法\n", errNum);
 				break;
 			case CL_INVALID_DEVICE_TYPE:
 				printf("errNum : %d, 传入的设备类型不是合法值\n", errNum);
 				break;
 			case CL_INVALID_PLATFORM:
 				printf("errNum : %d, 传入的平台不是合法值\n", errNum);
 				break;
 			case CL_INVALID_DEVICE:
 				printf("errNum : %d, 传入的设备不是合法值\n", errNum);
 				break;
 			case CL_INVALID_CONTEXT:
 				printf("errNum : %d, 传入的上下文不是合法值\n", errNum);
 				break;
 			case CL_INVALID_QUEUE_PROPERTIES:
 				printf("errNum : %d, 设备不支持命令队列属性\n", errNum);
 				break;
 			case CL_INVALID_COMMAND_QUEUE:
 				printf("errNum : %d, 传入的命令队列不是合法值\n", errNum);
 				break;
 			case CL_INVALID_HOST_PTR:
 				printf("errNum : %d, 主机指针不合法\n", errNum);
 				break;
 			case CL_INVALID_MEM_OBJECT:
 				printf("errNum : %d, 传入的内存对象不是合法值\n", errNum);
 				break;
 			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
 				printf("errNum : %d, 传入的图像格式描述符不是合法值\n", errNum);
 				break;
 			case CL_INVALID_IMAGE_SIZE:
 				printf("errNum : %d, 设备不支持这个图像的大小\n", errNum);
 				break;
 			case CL_INVALID_SAMPLER:
 				printf("errNum : %d, 传入的采样器不是合法值\n", errNum);
 				break;
 			case CL_INVALID_BINARY:
 				printf("errNum : %d, 传入了非法的二进制程序\n", errNum);
 				break;
 			case CL_INVALID_BUILD_OPTIONS:
 				printf("errNum : %d, 一个或多构建选项不合法\n", errNum);
 				break;
 			case CL_INVALID_PROGRAM:
 				printf("errNum : %d, 传入的程序不是合法值\n", errNum);
 				break;
 			case CL_INVALID_PROGRAM_EXECUTABLE:
 				printf("errNum : %d, 程序执行失败\n", errNum);
 				break;
 			case CL_INVALID_KERNEL_NAME:
 				printf("errNum : %d, 程序中不存在指定的内核\n", errNum);
 				break;
 			case CL_INVALID_KERNEL_DEFINITION:
 				printf("errNum : %d, 程序源代码中定义的内核不合法\n", errNum);
 				break;
 			case CL_INVALID_KERNEL:
 				printf("errNum : %d, 传入的内核不是合法值\n", errNum);
 				break;
 			case CL_INVALID_ARG_INDEX:
 				printf("errNum : %d, 参数索引指示的参数对于内核不合法\n", errNum);
 				break;
 			case CL_INVALID_ARG_VALUE:
 				printf("errNum : %d, 内核参数值为NULL\n", errNum);
 				break;
 			case CL_INVALID_ARG_SIZE:
 				printf("errNum : %d, 参数大小与参数数据类型不匹配\n", errNum);
 				break;
 			case CL_INVALID_KERNEL_ARGS:
 				printf("errNum : %d, 一个或多个内核未赋值\n", errNum);
 				break;
 			case CL_INVALID_WORK_DIMENSION:
 				printf("errNum : %d, 工作维度值不合法\n", errNum);
 				break;
 			case CL_INVALID_WORK_GROUP_SIZE:
 				printf("errNum : %d, 局部或全局工作组大小不合适\n", errNum);
 				break;
 			case CL_INVALID_WORK_ITEM_SIZE:
 				printf("errNum : %d, 一个或多个工作项大小超出了设备支持的最大值\n", errNum);
 				break;
 			case CL_INVALID_GLOBAL_OFFSET:
 				printf("errNum : %d, 全局偏移量超出了所支持的界限\n", errNum);
 				break;
 			case CL_INVALID_EVENT_WAIT_LIST:
 				printf("errNum : %d, 提供的等待时间大小不合法或者其中包含了非法事件\n", errNum);
 				break;
 			case CL_INVALID_EVENT:
 				printf("errNum : %d, 传入的事件不是一个合法值\n", errNum);
 				break;
 			case CL_INVALID_OPERATION:
 				printf("errNum : %d, 执行命令导致出现一个不合法的操作\n", errNum);
 				break;
 			case CL_INVALID_GL_OBJECT:
 				printf("errNum : %d, 不是一个有效的OpenGL内存对象\n", errNum);
 				break;
 			case CL_INVALID_BUFFER_SIZE:
 				printf("errNum : %d, 指定的缓冲区大小越界\n", errNum);
 				break;
 			case CL_INVALID_MIP_LEVEL:
 				printf("errNum : %d, 为OpenCL纹理指定的mipmap级别对于OpenGL对象不合法\n", errNum);
 				break;
 			case CL_INVALID_GLOBAL_WORK_SIZE:
 				printf("errNum : %d, 传入的全局工作大小不合法\n", errNum);
 				break;
 			default:
 				printf("程序运行不到这里...");
 				break;
 		}
 		return 0;
 	}
 }
void get_global_work_size1d(size_t n, size_t* global_work_size1d)
{
	size_t num_group_x = (n + local_work_size1d[0] - 1) / local_work_size1d[0];
	// printf("num_group_x %d\n", num_group_x);
	global_work_size1d[0] = local_work_size1d[0] * num_group_x;
}
void get_global_work_size2d(size_t x, size_t y, size_t* global_work_size2d)
{
	size_t num_group_x = (x + local_work_size2d[0] - 1) / local_work_size2d[0];
	global_work_size2d[0] = local_work_size2d[0] * num_group_x;
	size_t num_group_y = (y + local_work_size2d[1] - 1) / local_work_size2d[1];
	global_work_size2d[1] = local_work_size2d[1] * num_group_y;
}

void get_global_work_size3d(size_t x, size_t y, size_t z,  size_t* global_work_size3d)
{
	size_t num_group_x = (x + local_work_size3d[0] - 1) / local_work_size3d[0];
	global_work_size3d[0] = local_work_size2d[0] * num_group_x;
	size_t num_group_y = (y + local_work_size3d[1] - 1) / local_work_size3d[1];
	global_work_size3d[1] = local_work_size3d[1] * num_group_y;
	size_t num_group_z = (z + local_work_size3d[2] - 1) / local_work_size3d[2];
	global_work_size3d[2] = local_work_size3d[2] * num_group_z;
}
 cl_program createProgramWithSource(cl_context context, cl_device_id * devices, size_t device_num, char** kernel_name, size_t kernel_num, const char* options)
  {

	  cl_int errNum;
	  cl_program program;
	  char** kernel = (char**)malloc(kernel_num * sizeof(char*));
	  for(int i = 0; i < kernel_num; ++i){
	  		printf("%s\n", kernel_name[i]);
		  kernel[i] = read_kernel(kernel_name[i], NULL);
	  }
	  program =  clCreateProgramWithSource(context, kernel_num, (const char**)kernel, NULL, &errNum);
	  if(!check_errNum(errNum))
	  {
		  for(int i = 0; i < kernel_num; ++i)
			  free(kernel[i]);
		  free(kernel);
		  clReleaseProgram(program);
		  return NULL;
	  }
	  errNum = clBuildProgram(program, device_num, (const cl_device_id*)devices, options, NULL, NULL);
	  if(!check_errNum(errNum))
	  {
		  for(int i = 0; i < kernel_num; ++i)
			  free(kernel[i]);
		  free(kernel);
		  size_t LogSize;
		  char buffer[100];
		  FILE *fp = fopen("buildLog.txt", "wb");
		  if(NULL == fp)
		  {
			  printf("Fail to cteate buildLog.txt.\n");
			  for(int i = 0; i < kernel_num; ++i)
				  free(kernel[i]);
			  free(kernel);
			  clReleaseProgram(program);
			  return NULL;
		  }
		  for(int i=0; i < device_num; ++i)
		  {
			  clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 100, buffer, NULL);
			  clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &LogSize);
			  char* buildLog = (char*)malloc(LogSize * sizeof(char));
			  clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, sizeof(char) * LogSize, buildLog, NULL);
			  size_t device_name_length = strlen(buffer);
			  sprintf(buffer + device_name_length, " ,构建程序日志: " );
			  printf("%s(%d)\n%s\n\n", buffer,(int)LogSize,  buildLog);
			  fprintf(fp, "%s(%d)\n%s\n\n", buffer,(int)LogSize,  buildLog);
			  free(buildLog);
		  }
		  fclose(fp);
		  clReleaseProgram(program);
		  return NULL;
	  }
	  for(int i = 0; i < kernel_num; ++i)
		  free(kernel[i]);
	  free(kernel);
	  return program;
  }
cl_program createProgramWithBinary(cl_context context, cl_device_id device, const char* binary_path)
{
	cl_program program;
	cl_int errNum;
	cl_int binary_status;
	size_t binary_length;
	FILE* fp = fopen(binary_path, "rb");
	if (NULL == fp){
		printf("Fail to open %s\n", binary_path);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	binary_length = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	uchar* binary = (uchar*)malloc(binary_length * sizeof(uchar));
	fread(binary, sizeof(uchar), binary_length, fp);
	fclose(fp);
	program = clCreateProgramWithBinary (context, 1, &device, &binary_length, &binary, &binary_status, &errNum);
	if(!check_errNum(errNum))
	{
		free(binary);
		clReleaseProgram(program);
		return NULL;
	}

	if(!check_errNum(binary_status))
	{
		free(binary);
		clReleaseProgram(program);
		return NULL;
	}
	errNum = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if(!check_errNum(errNum))
	{
		free(binary);
		clReleaseProgram(program);
		return NULL;
	}
	free(binary);

	return program;
}
cl_mem cl_make_array(cl_context context, void* host_data, size_t size)
{
	cl_mem memObj;
	cl_int errNum;
	if(NULL == host_data)
	{
		memObj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &errNum);
		if(!check_errNum(errNum))
		{
			clReleaseMemObject(memObj);
			return NULL;
		}
		return memObj;
	}
	else
	{
		memObj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_data, &errNum);
		if(!check_errNum(errNum))
		{
			clReleaseMemObject(memObj);
			return NULL;
		}
		return memObj;
	}
}

cl_mem cl_push_array(cl_command_queue command, void* host_data, cl_mem device_data, size_t size)
{
	cl_int errNum;
	errNum = clEnqueueWriteBuffer(command, device_data, CL_TRUE, 0, size, host_data, 0, NULL, NULL);
	if(!check_errNum(errNum))
	{
		return NULL;
	}
	return device_data;

}
cl_mem cl_pull_array(cl_command_queue command, void* host_data, cl_mem device_data, size_t size)
{
	cl_int errNum;
	errNum = clEnqueueReadBuffer(command, device_data, CL_TRUE, 0, size, host_data, 0, NULL, NULL);
	if(!check_errNum(errNum))
	{
		return NULL;
	}
	return host_data;

}
 cl_context CreateContext(cl_device_id *device)
 {
     cl_int errNum;
     cl_uint numPlatforms;
     cl_context context = NULL;
     cl_platform_id FPGA_platform;
     errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
     printf("num of plats %d\n", numPlatforms);
     if (errNum != CL_SUCCESS || numPlatforms <= 0)
     {
         printf( "No platforms.\n" );
         return NULL;
     }
	 cl_platform_id *platform = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	 errNum = clGetPlatformIDs(numPlatforms, platform, NULL);
	 if (errNum != CL_SUCCESS)
	 {
		 printf("Failed to get platforms.\n");
		 free(platform);
		 return NULL;
	 }
	 size_t nm_sz;
         char name[256];
	 for (int i = 0; i < numPlatforms; ++i) {
		 errNum = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &nm_sz);
		 if (errNum != CL_SUCCESS)
		 {
			 printf("Failed to get platform name size.\n");
			 free(platform);
			 return NULL;
		 }
		 errNum = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, nm_sz, name, NULL);
		 if (errNum != CL_SUCCESS)
		 {
			 printf("Failed to get platform name.\n");
			 free(platform);
			 return NULL;
		 }
		 if (NULL != strstr(name, "Intel(R) FPGA")) {
			 FPGA_platform = platform[i];
			 free(platform);
			 break;
		 }

	 }
	 FPGA_platform = platform[0];
     errNum = clGetDeviceIDs(FPGA_platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
     if (errNum != CL_SUCCESS)
     {
        printf( "NO FPGAAA\n" );
         return NULL;
     }
     context = clCreateContext(NULL, 1, device, NULL, NULL, &errNum);
     if (errNum != CL_SUCCESS)
     {
         printf( "Fail to create context.\n" );
         return NULL;
     }
     return context;
 }

 cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device)
 {
     cl_command_queue commandQueue = NULL;
     commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
     if (commandQueue == NULL)
     {
         printf("Fail to create commandQueue.\n");
         return NULL;
     }
     return commandQueue;
 }

 void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3])
 {
     for (int i = 0; i < 3; i++)
     {
         if (memObjects[i] != 0)
             clReleaseMemObject(memObjects[i]);
     }
         if (commandQueue != 0)
             clReleaseCommandQueue(commandQueue);
         if (kernel != 0)
             clReleaseKernel(kernel);
         if (program != 0)
             clReleaseProgram(program);
         if (context != 0)
             clReleaseContext(context);
 }

#ifdef _WIN32 // Windows
 void *alignedMalloc(size_t size) {
	 return _aligned_malloc(size, DMA_ALIGNMENT);
 }
 void *alignedCalloc(size_t n, size_t size) {
	 void *p = _aligned_malloc(n * size, DMA_ALIGNMENT);
	 memset(p, 0, n * size);
	 return p;
 }
 void alignedFree(void * ptr) {
	 _aligned_free(ptr);
 }
#else          // Linux
 void *alignedMalloc(size_t size) {
	 void *result = NULL;
	 posix_memalign(&result, DMA_ALIGNMENT, size);
	 return result;
 }
 void *alignedCalloc(size_t n, size_t size) {
	 void *result = NULL;
	 posix_memalign(&result, DMA_ALIGNMENT, n*size);
	 memset(result, 0, n * size);
	 return result;
 }
 void alignedFree(void * ptr) {
	 free(ptr);
 }
#endif
