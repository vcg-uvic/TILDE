#include "CImageCl.h"

namespace SFOP {

COpenCl* CImageCl::s_opencl_p = NULL;

void CImageCl::initOpenCl()
{
    std::string kernelNames[] = {
        "downsample", "findLocalMax", "convRow", "convCol",
        "hessian", "inverse", "gradient", "solver", "negDefinite", "filter",
        "triSqr", "lambda2", "triSqrAlpha", "triSum", "precision", "bestOmega"
    };
    s_opencl_p = new COpenCl("sfopKernels.cl", kernelNames, NUM_KERNELS);
    if (!s_opencl_p) {
        std::cerr << "Error in CImageCl::initOpenCl(): new COpenCl() failed." << std::endl;
        exit(1);
    }
}

cl_mem CImageCl::newMem(
        const size_t f_size) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_result_p = clCreateBuffer(s_opencl_p->context(), CL_MEM_READ_WRITE, f_size, NULL, &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCL::newMem(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
    return l_result_p;
}

void CImageCl::freeMem(
        cl_mem f_mem_p) const
{
    if (f_mem_p == NULL) {
        std::cerr << "Error in CImageCl::~CImageCl(): f_mem_p is NULL." << std::endl;
        exit(1);
    }
    cl_int l_err = clReleaseMemObject(f_mem_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::~CImageCl(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }
    f_mem_p = NULL;
}

CImageCl::CImageCl() :
    m_mem_p(NULL), m_width(0), m_height(0)
{
    if (!s_opencl_p) initOpenCl();
}

CImageCl::CImageCl(
        const cl_mem f_mem_p,
        const unsigned int f_width,
        const unsigned int f_height) :
    m_mem_p(f_mem_p), m_width(f_width), m_height(f_height)
{
    cl_int l_err = clRetainMemObject(f_mem_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::CImageCl(cl_mem, int, int): clRetainMemObject() failed." << std::endl;
        exit(1);
    }
}

CImageCl::CImageCl(
        const cimg_library::CImg<float>& f_img) :
    m_mem_p(NULL), m_width(f_img.width()), m_height(f_img.height())
{
    if (!s_opencl_p) initOpenCl();
    cl_int l_err = CL_SUCCESS;
    m_mem_p = clCreateBuffer(s_opencl_p->context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * m_width * m_height, (void*)f_img.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::CImageCl(CImg): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
}

CImageCl::CImageCl(
        const EFilterNames f_filterName,
        const float f_sigma) :
    m_mem_p(NULL), m_width(2 * (int)(4.0f * f_sigma) + 1), m_height(1)
{
    if (!s_opencl_p) initOpenCl();
    const unsigned short int l_kSize = 4.0f * f_sigma;
    cl_int l_err = CL_SUCCESS;
    m_mem_p = newMem(sizeof(cl_float) * m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FIL], 0, sizeof(short), (void*)&f_filterName);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FIL], 1, sizeof(float), (void*)&f_sigma);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FIL], 2, sizeof(short), (void*)&l_kSize);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FIL], 3, sizeof(cl_mem), (void*)&m_mem_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::CImageCl(EFilternames, float): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_globalWorksize_p = m_width;
    runKernel(FIL, &l_globalWorksize_p);
}

CImageCl::~CImageCl()
{
    freeMem(m_mem_p);
}

void CImageCl::load(const char f_filename[])
{
    cimg_library::CImg<float> l_loader;
    l_loader.load(f_filename);
    if (l_loader.spectrum() > 1) l_loader.RGBtoHSV().channel(2);
    while (l_loader.max() > 1) l_loader /= 255.0f;

    unsigned int modX = ROWS_BLOCKDIM_X * ROWS_RESULT_STEPS;
    unsigned int modY = COLUMNS_BLOCKDIM_Y * COLUMNS_RESULT_STEPS;
    m_width = ((l_loader.width() - 1) / modX + 1) * modX;
    m_height = ((l_loader.height() - 1) / modY + 1) * modY;
    l_loader.crop(0, 0, m_width - 1, m_height - 1, true);

    cl_int l_err = CL_SUCCESS;
    m_mem_p = clCreateBuffer(s_opencl_p->context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * m_width * m_height, (void*)l_loader.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::load(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
}

void CImageCl::runKernel(
        const unsigned int f_kernel,
        const size_t* f_globalWorksize_p,
        const size_t* f_localWorksize_p,
        const unsigned int f_dim) const
{
    const size_t l_defaultSize = 1;
    const size_t* l_localWorksize_p  = f_localWorksize_p  ? f_localWorksize_p  : &l_defaultSize;
    const size_t* l_globalWorksize_p = f_globalWorksize_p ? f_globalWorksize_p : &l_defaultSize;
    cl_int l_err = clEnqueueNDRangeKernel(
            s_opencl_p->queue(), s_opencl_p->kernels()[f_kernel],
            f_dim, NULL, l_globalWorksize_p, l_localWorksize_p, 0, NULL, NULL);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::runKernel(): clEnqueueNDRangeKernel() failed." << std::endl;
        exit(1);
    }
    l_err = clFinish(s_opencl_p->queue());
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::runKernel(): clFinish() failed." << std::endl;
        exit(1);
    }
}

CImage* CImageCl::conv(
        const CImage *f_rowFilter_p,
        const CImage *f_colFilter_p)
{
    cl_int l_rowRadius = f_rowFilter_p->width() * f_rowFilter_p->height() / 2;
    cl_int l_colRadius = f_colFilter_p->width() * f_colFilter_p->height() / 2;

    cl_mem l_tmp_p = newMem(sizeof(cl_float) * m_width * m_height); 
    cl_mem l_rowFilter_mask_p = dynamic_cast<const CImageCl*>(f_rowFilter_p)->mem_p();
    cl_mem l_colFilter_mask_p = dynamic_cast<const CImageCl*>(f_colFilter_p)->mem_p();

    // run first kernel
    size_t l_localWorkSizeRow_p[2] = {ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y};
    size_t l_globalWorkSizeRow_p[2] = {m_width / ROWS_RESULT_STEPS, m_height};
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 0, sizeof(cl_mem), (void*)&l_tmp_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 1, sizeof(cl_mem), (void*)&m_mem_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 2, sizeof(cl_mem), (void*)&l_rowFilter_mask_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 3, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 4, sizeof(cl_int), (void*)&m_height);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 5, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVR], 6, sizeof(cl_int), (void*)&l_rowRadius);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::conv(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(CONVR, l_globalWorkSizeRow_p, l_localWorkSizeRow_p, 2);
    
    // run second kernel
    size_t l_localWorkSizeCol_p[2] = {COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y};
    size_t l_globalWorkSizeCol_p[2] = {m_width, m_height / COLUMNS_RESULT_STEPS};
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 0, sizeof(cl_mem), (void*)&m_mem_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 1, sizeof(cl_mem), (void*)&l_tmp_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 2, sizeof(cl_mem), (void*)&l_colFilter_mask_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 3, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 4, sizeof(cl_int), (void*)&m_height);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 5, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[CONVC], 6, sizeof(cl_int), (void*)&l_colRadius);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::conv(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(CONVC, l_globalWorkSizeCol_p, l_localWorkSizeCol_p, 2);

    // clean up
    l_err = clReleaseMemObject(l_tmp_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::conv(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }

    return this;
}

void CImageCl::triSqr(
        const CImage* f_gx_p,
        const CImage* f_gy_p,
        CImage* &f_gx2_p,
        CImage* &f_gxy_p,
        CImage* &f_gy2_p) const
{
    unsigned int l_width  = f_gx_p->width();
    unsigned int l_height = f_gx_p->height();
    cl_mem l_gx_p = dynamic_cast<const CImageCl*>(f_gx_p)->mem_p();
    cl_mem l_gy_p = dynamic_cast<const CImageCl*>(f_gy_p)->mem_p();
    cl_mem l_gx2_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_mem l_gxy_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_mem l_gy2_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQR], 0, sizeof(cl_mem), (void*)&l_gx_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQR], 1, sizeof(cl_mem), (void*)&l_gy_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQR], 2, sizeof(cl_mem), (void*)&l_gx2_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQR], 3, sizeof(cl_mem), (void*)&l_gxy_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQR], 4, sizeof(cl_mem), (void*)&l_gy2_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::triSqr(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(TRISQR, &l_global_size);
    f_gx2_p = new CImageCl(l_gx2_p, l_width, l_height);
    f_gxy_p = new CImageCl(l_gxy_p, l_width, l_height);
    f_gy2_p = new CImageCl(l_gy2_p, l_width, l_height);
    freeMem(l_gx2_p);
    freeMem(l_gxy_p);
    freeMem(l_gy2_p);
}

CImage* CImageCl::lambda2(
        const float f_M,
        const CImage* f_Nxx_p,
        const CImage* f_Nxy_p,
        const CImage* f_Nyy_p) const
{
    unsigned int l_width  = f_Nxx_p->width();
    unsigned int l_height = f_Nxx_p->height();
    cl_mem l_Nxx_p = dynamic_cast<const CImageCl*>(f_Nxx_p)->mem_p();
    cl_mem l_Nxy_p = dynamic_cast<const CImageCl*>(f_Nxy_p)->mem_p();
    cl_mem l_Nyy_p = dynamic_cast<const CImageCl*>(f_Nyy_p)->mem_p();
    cl_mem l_lambda2_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[LAMBDA2], 0, sizeof(float),  (void*)&f_M);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[LAMBDA2], 1, sizeof(cl_mem), (void*)&l_Nxx_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[LAMBDA2], 2, sizeof(cl_mem), (void*)&l_Nxy_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[LAMBDA2], 3, sizeof(cl_mem), (void*)&l_Nyy_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[LAMBDA2], 4, sizeof(cl_mem), (void*)&l_lambda2_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::lambda2(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(LAMBDA2, &l_global_size);
    CImageCl* l_result_p = new CImageCl(l_lambda2_p, l_width, l_height);
    freeMem(l_lambda2_p);
    return l_result_p;
}

void CImageCl::triSqrAlpha(
        const float f_alpha,
        const CImage* f_gx_p,
        const CImage* f_gy_p,
        CImage* &f_gx2a_p,
        CImage* &f_2gxya_p,
        CImage* &f_gy2a_p) const
{
    unsigned int l_width  = f_gx_p->width();
    unsigned int l_height = f_gx_p->height();
    cl_mem l_gx_p = dynamic_cast<const CImageCl*>(f_gx_p)->mem_p();
    cl_mem l_gy_p = dynamic_cast<const CImageCl*>(f_gy_p)->mem_p();
    cl_mem l_gx2a_p  = newMem(sizeof(cl_float) * l_width * l_height);
    cl_mem l_2gxya_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_mem l_gy2a_p  = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 0, sizeof(float),  (void*)&f_alpha);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 1, sizeof(cl_mem), (void*)&l_gx_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 2, sizeof(cl_mem), (void*)&l_gy_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 3, sizeof(cl_mem), (void*)&l_gx2a_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 4, sizeof(cl_mem), (void*)&l_2gxya_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISQRALPHA], 5, sizeof(cl_mem), (void*)&l_gy2a_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::triSqrAlpha(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(TRISQRALPHA, &l_global_size);
    f_gx2a_p  = new CImageCl(l_gx2a_p, l_width, l_height);
    f_2gxya_p = new CImageCl(l_2gxya_p, l_width, l_height);
    f_gy2a_p  = new CImageCl(l_gy2a_p, l_width, l_height);
    freeMem(l_gx2a_p);
    freeMem(l_2gxya_p);
    freeMem(l_gy2a_p);
}

CImage* CImageCl::triSum(
        const CImage* f_a_p,
        const CImage* f_b_p,
        const CImage* f_c_p) const
{
    unsigned int l_width  = f_a_p->width();
    unsigned int l_height = f_a_p->height();
    cl_mem l_a_p = dynamic_cast<const CImageCl*>(f_a_p)->mem_p();
    cl_mem l_b_p = dynamic_cast<const CImageCl*>(f_b_p)->mem_p();
    cl_mem l_c_p = dynamic_cast<const CImageCl*>(f_c_p)->mem_p();
    cl_mem l_sum_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISUM], 0, sizeof(cl_mem), (void*)&l_a_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISUM], 1, sizeof(cl_mem), (void*)&l_b_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISUM], 2, sizeof(cl_mem), (void*)&l_c_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[TRISUM], 3, sizeof(cl_mem), (void*)&l_sum_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::triSum(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(TRISUM, &l_global_size);
    CImageCl* l_result_p = new CImageCl(l_sum_p, l_width, l_height);
    freeMem(l_sum_p);
    return l_result_p;
}

CImage* CImageCl::precision(
        const float f_factor,
        const CImage* f_lambda2_p,
        const CImage* f_omega_p) const
{
    unsigned int l_width  = f_lambda2_p->width();
    unsigned int l_height = f_lambda2_p->height();
    cl_mem l_lambda2_p = dynamic_cast<const CImageCl*>(f_lambda2_p)->mem_p();
    cl_mem l_omega_p   = dynamic_cast<const CImageCl*>(f_omega_p  )->mem_p();
    cl_mem l_precision_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[PRECISION], 0, sizeof(float),  (void*)&f_factor);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[PRECISION], 1, sizeof(cl_mem), (void*)&l_lambda2_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[PRECISION], 2, sizeof(cl_mem), (void*)&l_omega_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[PRECISION], 3, sizeof(cl_mem), (void*)&l_precision_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::precision(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(PRECISION, &l_global_size);
    CImageCl* l_result_p = new CImageCl(l_precision_p, l_width, l_height);
    freeMem(l_precision_p);
    return l_result_p;
}

CImage* CImageCl::bestOmega(
        const CImage* f_omega0_p,
        const CImage* f_omega60_p,
        const CImage* f_omega120_p) const
{
    unsigned int l_width  = f_omega0_p->width();
    unsigned int l_height = f_omega0_p->height();
    cl_mem l_omega0_p   = dynamic_cast<const CImageCl*>(f_omega0_p)->mem_p();
    cl_mem l_omega60_p  = dynamic_cast<const CImageCl*>(f_omega60_p)->mem_p();
    cl_mem l_omega120_p = dynamic_cast<const CImageCl*>(f_omega120_p)->mem_p();
    cl_mem l_omegaMin_p = newMem(sizeof(cl_float) * l_width * l_height);
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[BESTOMEGA], 0, sizeof(cl_mem), (void*)&l_omega0_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[BESTOMEGA], 1, sizeof(cl_mem), (void*)&l_omega60_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[BESTOMEGA], 2, sizeof(cl_mem), (void*)&l_omega120_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[BESTOMEGA], 3, sizeof(cl_mem), (void*)&l_omegaMin_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::bestOmega(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_global_size = (size_t) l_width * l_height;
    runKernel(BESTOMEGA, &l_global_size);
    CImageCl* l_result_p = new CImageCl(l_omegaMin_p, l_width, l_height);
    freeMem(l_omegaMin_p);
    return l_result_p;
}

CImage* CImageCl::downsample()
{
    unsigned int l_oldWidth  = m_width;
    unsigned int l_oldHeight = m_height;

    unsigned int modX = ROWS_BLOCKDIM_X * ROWS_RESULT_STEPS;
    unsigned int modY = COLUMNS_BLOCKDIM_Y * COLUMNS_RESULT_STEPS;
    m_width  = ((m_width  / 2 - 1) / modX + 1) * modX;
    m_height = ((m_height / 2 - 1) / modY + 1) * modY;

    cl_int l_err = CL_SUCCESS;
    cl_mem l_result_p = newMem(sizeof(cl_float) * m_width * m_height);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DOWN], 0, sizeof(cl_mem), (void*)&m_mem_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DOWN], 1, sizeof(cl_int), (void*)&l_oldWidth);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DOWN], 2, sizeof(cl_int), (void*)&l_oldHeight);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DOWN], 3, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DOWN], 4, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::downsample(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_localWorksize_p[2] = {1, 1};
    size_t l_globalWorksize_p[2] = {m_width, m_height};
    runKernel(DOWN, l_globalWorksize_p, l_localWorksize_p, 2);

    freeMem(m_mem_p);
    m_mem_p = l_result_p;
    return this;
}

CImage* CImageCl::clone() const
{
    cl_mem l_copy_p = newMem(sizeof(cl_float) * m_width * m_height);
    cl_int l_err = clEnqueueCopyBuffer(s_opencl_p->queue(), m_mem_p, l_copy_p, 0, 0,
            (size_t) (sizeof(cl_float) * m_width * m_height), 0, NULL, NULL);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::clone(): clEnqueueCopyBuffer() failed." << std::endl;
        exit(1);
    }
    l_err = clFinish(s_opencl_p->queue());
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::clone(): clFinish() failed." << std::endl;
        exit(1);
    }

    CImageCl* l_result_p = new CImageCl(l_copy_p, m_width, m_height);
    freeMem(l_copy_p);
    return l_result_p;
}

cimg_library::CImg<float> CImageCl::computeGradient(
        const cimg_library::CImg<float> &f_cube) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_matrix_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3 * 3 * 3, (void*)f_cube.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeGradient(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
    cl_mem l_result_p = newMem(sizeof(cl_float) * 3);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[GRD], 0, sizeof(cl_mem), (void*)&l_matrix_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[GRD], 1, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeGradient(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(GRD);

    l_err = clReleaseMemObject(l_matrix_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeGradient(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }

    CImageCl l_result = CImageCl(l_result_p, 1, 3);
    freeMem(l_result_p);
    return l_result.asCimg();    
}

cimg_library::CImg<float> CImageCl::computeHessian(
        const cimg_library::CImg<float> &f_cube) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_matrix_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3 * 3 * 3, (void*)f_cube.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeHessian(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
    cl_mem l_result_p = newMem(sizeof(cl_float) * 3 * 3);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[HES], 0, sizeof(cl_mem), (void*)&l_matrix_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[HES], 1, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeHessian(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(HES);

    l_err = clReleaseMemObject(l_matrix_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeHessian(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }

    CImageCl l_result = CImageCl(l_result_p, 3, 3);
    freeMem(l_result_p);
    return l_result.asCimg();    
}

cimg_library::CImg<float> CImageCl::computeInverse(
        const cimg_library::CImg<float> &f_matrix) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_matrix_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3 * 3, (void*)f_matrix.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeInverse(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
    cl_mem l_result_p = newMem(sizeof(cl_float) * 3 * 3);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[INV], 0, sizeof(cl_mem), (void*)&l_matrix_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[INV], 1, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeInverse(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(INV);

    l_err = clReleaseMemObject(l_matrix_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeInverse(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }

    CImageCl l_result = CImageCl(l_result_p, 3, 3);
    freeMem(l_result_p);
    return l_result.asCimg();    
}

cimg_library::CImg<float> CImageCl::computeSolver(
        const cimg_library::CImg<float> &f_matrix,
        const cimg_library::CImg<float> &f_vector) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_matrix_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3 * 3, (void*)f_matrix.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeSolver(): clCreateBuffer() (matrix) failed." << std::endl;
        exit(1);
    }
    cl_mem l_vector_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3, (void*)f_vector.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeSolver(): clCreateBuffer() (vector) failed." << std::endl;
        exit(1);
    }
    cl_mem l_result_p = newMem(sizeof(cl_float) * 3);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[SLV], 0, sizeof(cl_mem), (void*)&l_matrix_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[SLV], 1, sizeof(cl_mem), (void*)&l_vector_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[SLV], 2, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeSolver(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(SLV);

    l_err = clReleaseMemObject(l_matrix_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeSolver(): clReleaseMemObject() (matrix) failed." << std::endl;
        exit(1);
    }
    l_err = clReleaseMemObject(l_vector_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeSolver(): clReleaseMemObject() (vector) failed." << std::endl;
        exit(1);
    }

    CImageCl l_result = CImageCl(l_result_p, 1, 3);
    freeMem(l_result_p);
    return l_result.asCimg();    
}

bool CImageCl::computeIfNegDefinite(
        const cimg_library::CImg<float> &f_matrix) const
{
    cl_int l_err = CL_SUCCESS;
    cl_mem l_matrix_p = clCreateBuffer(s_opencl_p->context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * 3 * 3, (void*)f_matrix.data(), &l_err);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeIfNegDefinite(): clCreateBuffer() failed." << std::endl;
        exit(1);
    }
    cl_mem l_result_p = newMem(sizeof(cl_float));
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DEF], 0, sizeof(cl_mem), (void*)&l_matrix_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[DEF], 1, sizeof(cl_mem), (void*)&l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeIfNegDefinite(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    runKernel(DEF);

    l_err = clReleaseMemObject(l_matrix_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::computeIfNegDefinite(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }

    cimg_library::CImg<float> l_cimgResult = CImageCl(l_result_p, 1, 1).asCimg();    
    freeMem(l_result_p);
    return l_cimgResult(0) != 0.0f;
}

cimg_library::CImg<float> CImageCl::asCimg() const
{
    cimg_library::CImg<float> l_cimg =
        cimg_library::CImg<float>(m_width, m_height);
    cl_int l_err = clEnqueueReadBuffer(s_opencl_p->queue(), m_mem_p, CL_TRUE, 0,
            sizeof(cl_float) * m_width * m_height, l_cimg.data(), 0, NULL, NULL);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::asCimg(): clEnqueueReadBuffer() failed." << std::endl;
        exit(1);
    }
    return l_cimg;
}

void CImageCl::findLocalMax(
        std::list<CFeature>* f_features_p,
        const CImage* f_below_p,
        const CImage* f_above_p,
        const CImage* f_lambda2_p,
        const float f_sigma,
        const unsigned int f_numLayers,
        const unsigned int f_factor) const
{
    unsigned int blockWidth = FLM_L_WG_S_X;
    unsigned int blockHeight = FLM_L_WG_S_Y;
    size_t sharedMemSize = (blockWidth + 2) * (blockHeight + 2) * sizeof(cl_float);
    cl_mem l_below_p  = (dynamic_cast <const CImageCl*> (f_below_p))->mem_p();
    cl_mem l_above_p  = (dynamic_cast <const CImageCl*> (f_above_p))->mem_p();
    cl_mem l_result_p = newMem(sizeof(cl_float4) * m_width * m_height);

    // run kernel with the 3 layers
    cl_int l_err = CL_SUCCESS;
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  0, sizeof(cl_mem), (void*)&l_below_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  1, sizeof(cl_mem), (void*)&m_mem_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  2, sizeof(cl_mem), (void*)&l_above_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  3, sizeof(cl_mem), (void*)&l_result_p);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  4, sizeof(cl_int), (void*)&m_width);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  5, sizeof(cl_int), (void*)&m_height);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  6, sharedMemSize, NULL); 
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  7, sharedMemSize, NULL); 
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  8, sharedMemSize, NULL); 
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM],  9, sizeof(cl_int), (void*)&blockWidth);
    l_err |= clSetKernelArg(s_opencl_p->kernels()[FLM], 10, sizeof(cl_int), (void*)&blockHeight);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::findLocalMax(): clSetKernelArg() failed." << std::endl;
        exit(1);
    }
    size_t l_workGroupSize[2] = {FLM_L_WG_S_X, FLM_L_WG_S_Y};
    size_t l_numWorkItems[2] = {
            m_width  + (l_workGroupSize[0] - m_width  % l_workGroupSize[0]),
            m_height + (l_workGroupSize[1] - m_height % l_workGroupSize[1])};
    runKernel(FLM, l_numWorkItems, l_workGroupSize, 2);

    // copy result float4 layer with the information for the maximas back to the cpu ram
    cl_float4* l_resultCpu_p = (cl_float4*) malloc(sizeof(cl_float4) * m_width * m_height);
    if (l_resultCpu_p == NULL) {
        std::cerr << "Error in CImageCl::findLocalMax(): malloc() failed." << std::endl;
        exit(1);
    }
    l_err = clEnqueueReadBuffer(s_opencl_p->queue(), l_result_p, CL_TRUE, 0,
            sizeof(cl_float4) * m_width * m_height, (void*)l_resultCpu_p, 0, NULL, NULL);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::findLocalMax(): clEnqueueReadBuffer() failed." << std::endl;
        exit(1);
    }

    // scan through 1st result layer to identify local maxima and build up a list
    const cimg_library::CImg<float> l_lambda2 = f_lambda2_p->asCimg();
    cimg_for_insideXY(l_lambda2, x, y, 1) {
        const unsigned int l_idx = x + y * m_width;
        if (l_resultCpu_p[l_idx].s[3] == 0.0f) continue;
        const float l_x = l_resultCpu_p[l_idx].s[0];
        const float l_y = l_resultCpu_p[l_idx].s[1];
        const float l_s = l_resultCpu_p[l_idx].s[2];
        const float l_p = l_resultCpu_p[l_idx].s[3];
        const float l_sigma = f_sigma * pow(2.0f, l_s / f_numLayers);
        const float l_l2 = l_lambda2(floor(l_x + 0.5f), floor(l_y + 0.5f));
        f_features_p->push_back(CFeature(
                    l_x * (float) f_factor,
                    l_y * (float) f_factor,
                    l_sigma * (float) f_factor,
                    l_l2 / (float) f_factor / (float) f_factor,
                    l_p * (float) f_factor * (float) f_factor));
    }

    l_err = clReleaseMemObject(l_result_p);
    if (!s_opencl_p->checkErr(l_err)) {
        std::cerr << "Error in CImageCl::findLocalMax(): clReleaseMemObject() failed." << std::endl;
        exit(1);
    }
    free(l_resultCpu_p);
}

}

