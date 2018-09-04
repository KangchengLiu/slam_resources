#include "stereo_bm_cl.h"

OpenCLObjects openCLObjects;

cl_mem leftBuffer, rightBuffer, leftRightResultBuffer, leftRightCopyBuffer, rightLeftBuffer;

void init_ocl(){
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    OPENCLinitialization (Platform, Device, program, etc.)
    */
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int idP=-1;
    cl_uint num_platforms = 0, numDevices = 0, i = 0;
    cl_int errCheck = 0;
    size_t platform_name_length = 0, workitem_size[3], workgroup_size, address_bits;

    /* Step 1 */
    /*	Query for all available OpenCL platforms and devices on the system.
        Select a platform that has the required PREFERRED_PLATFORM substring using strstr-function.
    */

    // Get total number of the available platforms.
    errCheck = clGetPlatformIDs(0, 0, &num_platforms);
    cl_errCheck(errCheck,"Platform inquiry",true);
    printf("Number of available platforms: %u \n", num_platforms);

    // Get IDs for all platforms.
    vector<cl_platform_id> platforms(num_platforms);
    errCheck = clGetPlatformIDs(num_platforms, &platforms[0], 0);
    cl_errCheck(errCheck,"clGetPlatformIds",true);

    // Get the size of the platform name in bytes.
    errCheck = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &platform_name_length);
    cl_errCheck(errCheck,"clGetPlatformInfo",true);

    for (i=0; i<num_platforms; i++) {

        // Get the actual name for the i-th platform.
        vector<char> platform_name(platform_name_length);
        errCheck = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, &platform_name[i], 0);
        cl_errCheck(errCheck, "clGetPlatformIInfo Names", true);

        //Print out the platform id and name
        string platforName = &platform_name[i];
        printf("\n[%u] %s \n", i, (platforName).c_str());

        // Check if the platform is the preferred platform
        if (strstr(&platform_name[i], PREFERRED_PLATFORM)) {
            openCLObjects.platform = platforms[i];
            idP = i;
        }

/////////////// This is where the exercise work begins //////////////////////////////////////////////////////////
        /* STEP 2 */
        /* All the platforms are now queried and selected. Next you need to query all the available devices for the platforms.
            Of course you can only query the devices for the selected platform and then select a suitable device.
            Depending on your approach place the device query inside or outside of the above for loop. The reason for scanning all
            the devices in each platform is just to show you what device options you might have.
        */
    }

    if (idP == -1) {
        printf("Preferred platform not found. Exiting...");
        exit(1);
    } else
        printf("\nPlaform ID [%u] selected \n", idP);

    //find the devices and print how many devices found + device information
    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platforms[idP], CL_DEVICE_TYPE_ALL, sizeof(openCLObjects.device), &openCLObjects.device, &deviceIdCount);
    cout << "Found " << deviceIdCount << " device(s)" << endl;

    char deviceName[1024];
    cl_ulong local_mem_size;
    clGetDeviceInfo(openCLObjects.device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(openCLObjects.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    cout << "Name: " << deviceName << endl;
    cout << "Local memory size: " << local_mem_size << endl;

    /* STEP 3 */
    /* You have now selected a platform and a device to use either a CPU or a GPU in our case. In this exercise we simply use one device at a
    time. Of course you are free to implement a multidevice setup if you want. Now you need to create context for the selected device.
    */
    const cl_context_properties contextProperties [] =
            {
                    CL_CONTEXT_PLATFORM,
                    reinterpret_cast<cl_context_properties>(platforms[idP]),
                    0, 0
            };

    //initialize variable for error
    cl_int error = CL_SUCCESS;

    //create context
    openCLObjects.context = clCreateContext(contextProperties, deviceIdCount, &openCLObjects.device, nullptr, nullptr, &error);

    /* STEP 4. */
    /* Query for the OpenCL device that was used for context creation using clGetContextInfo. This step is just to check
        that no errors occured during context creation step. Error handling is very important in OpenCL since the bug might be in the host or
        kernel code. Use the errCheck-function on every step to indicate the location of the possible bug.
    */

    int max_work_group;

    //initialize a test variable that is only used for querying the context
    cl_device_id contextsDevice;
    cl_errCheck(clGetContextInfo(openCLObjects.context, CL_CONTEXT_DEVICES, sizeof(contextsDevice), &contextsDevice, NULL),
                "context creation failed",
                true);
    cout << "Context created successfully";
    clGetDeviceInfo(contextsDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    cout << " for: " << deviceName << endl;
    clGetDeviceInfo(contextsDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group), &max_work_group, NULL);
    cout << "max workgroup size in bytes: " << max_work_group << endl;

    /* STEP 5. */
    /*	Create OpenCL program from the kernel file source code
        First read the source kernel code in from the .cl file as an array of char's.
        Then "submit" the source code of the kernel to OpenCL and create a program object with it.
    */

    //create kernel program object
    openCLObjects.program = CreateProgram ("/home/cg/projects/visual_slam/Mapping/ros_wrapper/ssd_block_match.cl", openCLObjects.context);
    cout << "Program object created" << endl;

    /* STEP 6. */
    /* Build the program. The program is now created but not built. Next you need to build it
    */

    cl_errCheck(clBuildProgram (openCLObjects.program, deviceIdCount, &openCLObjects.device, nullptr, nullptr, nullptr),
                "Program building failed",
                true);
    cout << "Program built" << endl;

    /* STEP 7. */
    /*	Extract the kernel/kernels from the built program. The program consists of one or more kernels. Each kernel needs to be enqueued for
        execution from the host code. Creating a kernel via clCreateKernel is similar to obtaining an entry point of a specific function
        in an OpenCL program.
    */

    openCLObjects.kernel[0] = clCreateKernel (openCLObjects.program, "SSD_depth_estimator", &error);
    cl_errCheck(error,"Kernel creation failed", true);
    cout << "Kernel created" << endl;

    /* STEP 8. */
    /*	Now that you have created the kernel/kernels, you also need to enqueue them for execution to the selected device. The command queu can
    be either in-order or out-of-order type depending on your approach. In-order command queue is a good starting point.
    */
    openCLObjects.queue = clCreateCommandQueue(openCLObjects.context, openCLObjects.device, CL_QUEUE_PROFILING_ENABLE, &error);
    cl_errCheck(error,"Command queue creation failed",true);
    cout << "Created command queue" << endl;
}

void stereo_bm(unsigned char *imgL, unsigned char *imgR, unsigned char **resultL, int width, int height)
{
    if (imgL == NULL || imgR == NULL) {
        fprintf(stderr, "Memory allocation error!\n");
        exit(1);
    }

    int imSize = width * height;

    int win_size = 15;              //window dimension for SSD block matching e.g. 15 creates 15x15 window
    int max_disparity = 90;
    int l_width  = 32;              //local work group width
    int l_height = 16;              //local work group height

    //round up image width and height for work group sizes
    int pow2_width  = roundUp2(l_width, width);
    int pow2_height = roundUp2(l_height, height);

    //Remember that the global work group size needs to be a multiple of the local workgroup size.
    size_t globalWorkSize [2] = { pow2_width, pow2_height };
    size_t localWorkSize [2]  = { l_width, l_height };

    cout << "Global work size: " << globalWorkSize[0] << "x" << globalWorkSize[1] << endl;
    cout << "Local work size: " << localWorkSize[0] << "x" << localWorkSize[1] << endl;

    cl_int error = CL_SUCCESS;


    /* STEP 9. */
    /* Allocate device memory. You need to at least allocate memory on the device for the input and output images.
    Remember the error handling for the memory objects also.
    */
    static bool init_buf = false;

    if(!init_buf) {
        init_buf = true;

        size_t mem_size = sizeof(unsigned char) * imSize;

        leftBuffer = clCreateBuffer(openCLObjects.context, CL_MEM_READ_ONLY, mem_size, NULL, &error);
        cl_errCheck(error, "Memory buffer 1 creation failed", true);
        cout << "Created buffer for left input image" << endl;

        rightBuffer = clCreateBuffer(openCLObjects.context, CL_MEM_READ_ONLY, mem_size, NULL, &error);
        cl_errCheck(error, "Memory buffer 2 creation failed", true);
        cout << "Created buffer for right input image" << endl;

        leftRightResultBuffer = clCreateBuffer(openCLObjects.context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
        cl_errCheck(error, "Memory buffer 3 creation failed", true);
        cout << "Created buffer for left-right result image" << endl;

        leftRightCopyBuffer = clCreateBuffer(openCLObjects.context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
        cl_errCheck(error, "Memory buffer 4 creation failed", true);
        cout << "Created buffer for second left-right result image" << endl;

        rightLeftBuffer = clCreateBuffer(openCLObjects.context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
        cl_errCheck(error, "Memory buffer 5 creation failed", true);
        cout << "Created buffer for right-left result image" << endl;
    }

    clEnqueueWriteBuffer(openCLObjects.queue, leftBuffer,  CL_TRUE, 0, sizeof(unsigned char)*imSize, imgL, NULL, nullptr, NULL);
    clEnqueueWriteBuffer(openCLObjects.queue, rightBuffer, CL_TRUE, 0, sizeof(unsigned char)*imSize, imgR, NULL, nullptr, NULL);
    cout << "Buffers enqueued" << endl;

    win_size = (win_size-1)/2;          //correct the window size to be the "radius" of the box

    cout << "Local memory needed: " << sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size)*2 << endl;

    int i = 0;
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftBuffer);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &rightBuffer);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftRightResultBuffer);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &leftRightCopyBuffer);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_mem), &rightLeftBuffer);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size), NULL);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (cl_char4)*(localWorkSize[0]+max_disparity+2*win_size)*(localWorkSize[1]+2*win_size), NULL);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &width);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &height);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &win_size);
    clSetKernelArg (openCLObjects.kernel[0], i++, sizeof (int), &max_disparity);

    //Queue the kernel for execution
    cl_event event[2];
    clEnqueueNDRangeKernel(openCLObjects.queue, openCLObjects.kernel[0], 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, &event[0]);
    cout << "Execution started" << endl;

    clFinish(openCLObjects.queue);

    clEnqueueReadBuffer(openCLObjects.queue, leftRightResultBuffer, CL_TRUE, 0, sizeof(unsigned char)*imSize, *resultL, 0, nullptr, &event[1]);
    cout << "Result read back to host" << endl;

    clWaitForEvents(2, event);

    /*get the time taken to execute the kernel*/
    cl_ulong time_start = 0, time_end = 0;
    cl_errCheck(clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL), "start time failed", false);
    cl_errCheck(clGetEventProfilingInfo(event[1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL), "end time failed", false);
    double total_time = (double)(time_end - time_start)/1000000000.0;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);

    cout << "The OpenCL-implementation took " << total_time << " seconds to execute\n" << endl;
}

void release_ocl(){
    clReleaseMemObject (leftRightResultBuffer);
    clReleaseMemObject (leftRightCopyBuffer);
    clReleaseMemObject (rightLeftBuffer);
    cout << "Result image buffers released" << endl;

    clReleaseMemObject (rightBuffer);
    cout << "Right input image buffer released" << endl;

    clReleaseMemObject (leftBuffer);
    cout << "Left input image buffer released" << endl;


    clReleaseCommandQueue (openCLObjects.queue);
    cout << "Command queue released" << endl;

    clReleaseContext (openCLObjects.context);
    cout << "Context released" << endl;

    clReleaseKernel (openCLObjects.kernel[0]);
    cout << "Kernel released" << endl;
    clReleaseProgram (openCLObjects.program);
    cout << "Program released" << endl;
}