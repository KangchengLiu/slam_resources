#ifndef PROJECT_STEREO_BM_CL_H
#define PROJECT_STEREO_BM_CL_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>

#include <CL/cl.h>
#include <vector>

using namespace std;

#define PREFERRED_PLATFORM "NVIDIA"

/* OpenCL objects declaration */
struct OpenCLObjects
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel[2];
};

inline int roundUp2(int groupSize, int globalSize) {
    int r = globalSize % groupSize;

    if (r == 0)
        return globalSize;
    else
        return globalSize + groupSize - r;
}

/* A function to check for error code as per cl_int returned by OpenCl
 * Parameter errCheck   Error value as cl_int
 * Parameter msg        User provided error message
 * Return               True if Error found, False otherwise */

inline int cl_errCheck(const cl_int errCheck, const char * msg, bool exitOnError)
{
    char *cl_error[64] = {
            "CL_SUCCESS",                         // 0
            "CL_DEVICE_NOT_FOUND",                //-1
            "CL_DEVICE_NOT_AVAILABLE",            //-2
            "CL_COMPILER_NOT_AVAILABLE",          //-3
            "CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
            "CL_OUT_OF_RESOURCES",                //-5
            "CL_OUT_OF_HOST_MEMORY",              //-6
            "CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
            "CL_MEM_COPY_OVERLAP",                //-8
            "CL_IMAGE_FORMAT_MISMATCH",           //-9
            "CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
            "CL_BUILD_PROGRAM_FAILURE",           //-11
            "CL_MAP_FAILURE",                     //-12
            "",                                   //-13
            "",                                   //-14
            "",                                   //-15
            "",                                   //-16
            "",                                   //-17
            "",                                   //-18
            "",                                   //-19
            "",                                   //-20
            "",                                   //-21
            "",                                   //-22
            "",                                   //-23
            "",                                   //-24
            "",                                   //-25
            "",                                   //-26
            "",                                   //-27
            "",                                   //-28
            "",                                   //-29
            "CL_INVALID_VALUE",                   //-30
            "CL_INVALID_DEVICE_TYPE",             //-31
            "CL_INVALID_PLATFORM",                //-32
            "CL_INVALID_DEVICE",                  //-33
            "CL_INVALID_CONTEXT",                 //-34
            "CL_INVALID_QUEUE_PROPERTIES",        //-35
            "CL_INVALID_COMMAND_QUEUE",           //-36
            "CL_INVALID_HOST_PTR",                //-37
            "CL_INVALID_MEM_OBJECT",              //-38
            "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
            "CL_INVALID_IMAGE_SIZE",              //-40
            "CL_INVALID_SAMPLER",                 //-41
            "CL_INVALID_BINARY",                  //-42
            "CL_INVALID_BUILD_OPTIONS",           //-43
            "CL_INVALID_PROGRAM",                 //-44
            "CL_INVALID_PROGRAM_EXECUTABLE",      //-45
            "CL_INVALID_KERNEL_NAME",             //-46
            "CL_INVALID_KERNEL_DEFINITION",       //-47
            "CL_INVALID_KERNEL",                  //-48
            "CL_INVALID_ARG_INDEX",               //-49
            "CL_INVALID_ARG_VALUE",               //-50
            "CL_INVALID_ARG_SIZE",                //-51
            "CL_INVALID_KERNEL_ARGS",             //-52
            "CL_INVALID_WORK_DIMENSION ",         //-53
            "CL_INVALID_WORK_GROUP_SIZE",         //-54
            "CL_INVALID_WORK_ITEM_SIZE",          //-55
            "CL_INVALID_GLOBAL_OFFSET",           //-56
            "CL_INVALID_EVENT_WAIT_LIST",         //-57
            "CL_INVALID_EVENT",                   //-58
            "CL_INVALID_OPERATION",               //-59
            "CL_INVALID_GL_OBJECT",               //-60
            "CL_INVALID_BUFFER_SIZE",             //-61
            "CL_INVALID_MIP_LEVEL",               //-62
            "CL_INVALID_GLOBAL_WORK_SIZE"};       //-63


    if(errCheck != CL_SUCCESS) {
        printf("OpenCL Error: %d %s %s\n", errCheck, (char *)(cl_error[-errCheck]), msg);

        if(exitOnError) {
            exit(-1);
        }
        return true;
    }
    return false;
}

inline cl_program CreateProgram (const char* name, cl_context context)
{
    std::ifstream in (name);
    std::string source ( (std::istreambuf_iterator<char> (in)), std::istreambuf_iterator<char> () );

    size_t lengths [1] = { source.size () };
    const char* sources [1] = { source.data () };

    cl_int error = 0;
    cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
    cl_errCheck(error,"Program object creation failed", true);

    return program;
}

void init_ocl();
void release_ocl();

void stereo_bm(unsigned char *imgL, unsigned char *imgR, unsigned char **resultL, int width, int height);

#endif //PROJECT_STEREO_BM_CL_H
