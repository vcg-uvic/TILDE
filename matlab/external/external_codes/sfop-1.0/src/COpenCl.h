#ifndef OPENCL_H
#define OPENCL_H

#include <CL/cl.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <sys/stat.h> 

namespace SFOP {

/**
 * @brief Class managing OpenCL access, i.e. queue, context, kernels and
 * program.
 *
 * The Constructor checks available devices and loads and compiles OpenCL code.
 * The Destructor releases context, queue, kernels and program.
 */
class COpenCl
{

    private:

        /// Program
        cl_program m_program;

        /// Number of kernels
        cl_uint m_numKernels;

        /// Context
        cl_context m_context;

        /// Command queue
        cl_command_queue m_queue;

        /// Kernels
        cl_kernel* m_kernels;

    public:

        /** 
         * @brief Check OpenCl error code: Assert success, otherwise print message.
         * 
         * @param[in] f_err OpenCl error code
         *
         * @returns True if no error occured
         */
        bool checkErr(const cl_int f_err);

        /** 
         * @brief Constructor
         *
         * Checks available devices and loads and compiles OpenCL code.
         * 
         * @param[in] f_filename OpenCL file containing kernel source codes
         * @param[in] f_kernelNames Kernel names as defined in OpenCL file
         * @param[in] f_numKernels Number of kernels, i.e. number of kernel names
         */
        COpenCl(const std::string f_filename, const std::string* f_kernelNames, cl_uint f_numKernels);

        /**
         * @brief Destructor
         *
         * Stops and releases OpenCL objects.
         */
        ~COpenCl();

        /** 
         * @brief Public access to command queue
         * 
         * @return  Command queue
         */
        cl_command_queue queue() {return m_queue;}

        /** 
         * @brief Public access to context
         * 
         * @return Context
         */
        cl_context context() {return m_context;}

        /** 
         * @brief Public access to kernels
         * 
         * @return Kernels
         */
        cl_kernel *kernels() {return m_kernels;}

};

};

#endif
