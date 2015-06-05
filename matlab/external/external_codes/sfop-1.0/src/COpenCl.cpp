#include "COpenCl.h"

namespace SFOP {

bool COpenCl::checkErr(const cl_int f_err)
{
    switch (f_err) {
        case CL_SUCCESS: return true;
        case CL_DEVICE_NOT_FOUND:                std::cerr << "Device not found\n";                    break;
        case CL_DEVICE_NOT_AVAILABLE:            std::cerr << "Device not available\n";                break;
        case CL_COMPILER_NOT_AVAILABLE:          std::cerr << "Compiler not available\n";              break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   std::cerr << "Memory object allocation failure\n";    break;
        case CL_OUT_OF_RESOURCES:                std::cerr << "Out of resources\n";                    break;
        case CL_OUT_OF_HOST_MEMORY:              std::cerr << "Out of host memory\n";                  break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:    std::cerr << "Profiling information not available\n"; break;
        case CL_MEM_COPY_OVERLAP:                std::cerr << "Memory copy overlap\n";                 break;
        case CL_IMAGE_FORMAT_MISMATCH:           std::cerr << "Image format mismatch\n";               break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:      std::cerr << "Image format not supported\n";          break;
        case CL_BUILD_PROGRAM_FAILURE:           std::cerr << "Program build failure\n";               break;
        case CL_MAP_FAILURE:                     std::cerr << "Map failure\n";                         break;
        case CL_INVALID_VALUE:                   std::cerr << "Invalid value\n";                       break;
        case CL_INVALID_DEVICE_TYPE:             std::cerr << "Invalid device type\n";                 break;
        case CL_INVALID_PLATFORM:                std::cerr << "Invalid platform\n";                    break;
        case CL_INVALID_DEVICE:                  std::cerr << "Invalid device\n";                      break;
        case CL_INVALID_CONTEXT:                 std::cerr << "Invalid context\n";                     break;
        case CL_INVALID_QUEUE_PROPERTIES:        std::cerr << "Invalid queue properties\n";            break;
        case CL_INVALID_COMMAND_QUEUE:           std::cerr << "Invalid command queue\n";               break;
        case CL_INVALID_HOST_PTR:                std::cerr << "Invalid host pointer\n";                break;
        case CL_INVALID_MEM_OBJECT:              std::cerr << "Invalid memory object\n";               break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: std::cerr << "Invalid image format descriptor\n";     break;
        case CL_INVALID_IMAGE_SIZE:              std::cerr << "Invalid image size\n";                  break;
        case CL_INVALID_SAMPLER:                 std::cerr << "Invalid sampler\n";                     break;
        case CL_INVALID_BINARY:                  std::cerr << "Invalid binary\n";                      break;
        case CL_INVALID_BUILD_OPTIONS:           std::cerr << "Invalid build options\n";               break;
        case CL_INVALID_PROGRAM:                 std::cerr << "Invalid program\n";                     break;
        case CL_INVALID_PROGRAM_EXECUTABLE:      std::cerr << "Invalid program executable\n";          break;
        case CL_INVALID_KERNEL_NAME:             std::cerr << "Invalid kernel name\n";                 break;
        case CL_INVALID_KERNEL_DEFINITION:       std::cerr << "Invalid kernel definition\n";           break;
        case CL_INVALID_KERNEL:                  std::cerr << "Invalid kernel\n";                      break;
        case CL_INVALID_ARG_INDEX:               std::cerr << "Invalid argument index\n";              break;
        case CL_INVALID_ARG_VALUE:               std::cerr << "Invalid argument value\n";              break;
        case CL_INVALID_ARG_SIZE:                std::cerr << "Invalid argument size\n";               break;
        case CL_INVALID_KERNEL_ARGS:             std::cerr << "Invalid kernel arguments\n";            break;
        case CL_INVALID_WORK_DIMENSION:          std::cerr << "Invalid work dimension\n";              break;
        case CL_INVALID_WORK_GROUP_SIZE:         std::cerr << "Invalid work group size\n";             break;
        case CL_INVALID_WORK_ITEM_SIZE:          std::cerr << "Invalid work item size\n";              break;
        case CL_INVALID_GLOBAL_OFFSET:           std::cerr << "Invalid global offset\n";               break;
        case CL_INVALID_EVENT_WAIT_LIST:         std::cerr << "Invalid event wait list\n";             break;
        case CL_INVALID_EVENT:                   std::cerr << "Invalid event\n";                       break;
        case CL_INVALID_OPERATION:               std::cerr << "Invalid operation\n";                   break;
        case CL_INVALID_GL_OBJECT:               std::cerr << "Invalid OpenGL object\n";               break;
        case CL_INVALID_BUFFER_SIZE:             std::cerr << "Invalid buffer size\n";                 break;
        case CL_INVALID_MIP_LEVEL:               std::cerr << "Invalid mip-map level\n";               break;
        default:                                 std::cerr << "Unknown error code: " << f_err << "\n"; break;
    }
    return false;
}

COpenCl::COpenCl(
        const std::string f_filename,
        const std::string* f_kernelNames,
        const cl_uint f_numKernels)
{
    cl_uint l_num_platforms;
    cl_platform_id* l_platform_ids; //Must be freed in the end!

    cl_int l_err = clGetPlatformIDs (0, NULL, &l_num_platforms);
    if (!checkErr(l_err)) exit(1);

    if (l_num_platforms == 0) {
        std::cout << "No OpenCL platform found!" << std::endl;
        exit(1);
    }

    // if there's a platform or more, make space for ID's
    if ((l_platform_ids = (cl_platform_id*) malloc(l_num_platforms * sizeof(cl_platform_id))) == NULL) {
        std::cout << "Failed to allocate memory for cl_platform ID's!" << std::endl;
        exit(1);
    }
    l_err = clGetPlatformIDs(l_num_platforms, l_platform_ids, NULL);
    if (!checkErr(l_err)) exit(1);

    // search for cpus
    cl_device_id l_cpu = NULL;
    cl_int l_err_cpu = clGetDeviceIDs(l_platform_ids[0], CL_DEVICE_TYPE_CPU, 1, &l_cpu, NULL);

    // search for gpus
    cl_device_id l_devices = NULL;
    cl_int l_err_gpu = clGetDeviceIDs(l_platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &l_devices, NULL);
    checkErr(l_err_gpu);
    if (l_err_gpu != CL_SUCCESS) {
        if (!checkErr(l_err_cpu)) exit(1);
        l_devices = l_cpu;
    }
    free(l_platform_ids);

    //cl_uint info = 0;
    //clGetDeviceInfo(l_devices,CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), (void*)&info, NULL);
    //cl_ulong info2 = 0;
    //clGetDeviceInfo(l_devices,CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), (void*)&info2, NULL); 

    // context for the identified CL device
    m_context = clCreateContext(0, 1, &l_devices, NULL, NULL, &l_err);
    if (!checkErr(l_err)) exit(1);

    // command queue (respective to the context and device)
    m_queue = clCreateCommandQueue(m_context, l_devices, 0, &l_err);
    if (!checkErr(l_err)) exit(1);

    // check if opencl kernel file exists http://www.techbytes.ca/techbyte103.html
    struct stat stFileInfo;
    if (stat(f_filename.c_str(), &stFileInfo)) {
        std::cerr << "OpenCL Kernel file missing. " << std::endl;
        exit(1);
    } 

    // read cl file into c-string of fixed length
    std::vector<char> *l_container = new std::vector<char>();
    char l_ch;
    std::ifstream l_file(f_filename.data());
    if (l_file) while(l_file.get(l_ch)) l_container->push_back(l_ch);
    l_file.close();
    l_container->push_back('\0');
    char* l_source = new char[l_container->size()];
    for (cl_uint i = 0; i < l_container->size(); ++i) {
        l_source[i] = (*l_container)[i];
    }
    delete l_container;
    l_container = NULL;

    // create and build CL program
    m_program = clCreateProgramWithSource(
            m_context, 1, (const char**)&l_source, NULL, &l_err);
    delete[] l_source;
    l_source = NULL;
    if (!checkErr(l_err)) exit(1);
    //char l_options[] = "";
    char l_options[] = "-w"; //Inhibit all warnings
    //char l_options[] = "-Werror"; //Transform all warnings to errors

    l_err = clBuildProgram(m_program, 0, NULL, l_options, NULL, NULL);
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(m_program, l_devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size + 1];
    // Second call to get the log
    clGetProgramBuildInfo(m_program, l_devices, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    std::cout << build_log << std::endl;
    delete[] build_log;
    if (!checkErr(l_err)) exit(1);

    // create kernels
    m_numKernels = f_numKernels;
    m_kernels = new cl_kernel[f_numKernels];
    for (cl_uint i = 0; i < f_numKernels; ++i) {
        m_kernels[i] = clCreateKernel(m_program, f_kernelNames[i].data(), &l_err);
    }
}

COpenCl::~COpenCl()
{
    clReleaseCommandQueue(m_queue);
    clReleaseContext(m_context);
    for (cl_uint i = 0; i < m_numKernels; ++i) clReleaseKernel(m_kernels[i]);
    delete[] m_kernels;
    m_kernels = NULL;
    clReleaseProgram(m_program);
}

}

