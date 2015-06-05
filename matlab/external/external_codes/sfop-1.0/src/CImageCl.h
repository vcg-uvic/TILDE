#ifndef CIMAGECL_H
#define CIMAGECL_H

#include "COpenCl.h"
#include "CL/cl.h"
#include "CImage.h"
#include "CImg.h"

#define ROWS_BLOCKDIM_X      32
#define ROWS_BLOCKDIM_Y       4
#define ROWS_RESULT_STEPS     8
#define COLUMNS_BLOCKDIM_X   32
#define COLUMNS_BLOCKDIM_Y    8
#define COLUMNS_RESULT_STEPS  8

#define DOWN              0 ///< downsample
#define FLM               1 ///< findLocalMax
#define FLM_L_WG_S_X      8 ///< Work group size x for findLocalMax
#define FLM_L_WG_S_Y      8 ///< Work group sizy y for findLocalMax
#define CONVR             2 ///< convRow
#define CONVC             3 ///< convCol
#define HES               4 ///< hessian
#define INV               5 ///< inverse
#define GRD               6 ///< gradient
#define SLV               7 ///< solver
#define DEF               8 ///< negDefinite
#define FIL               9 ///< filter
#define TRISQR           10 ///< triSqr
#define LAMBDA2          11 ///< lambda2
#define TRISQRALPHA      12 ///< triSqrAlpha
#define TRISUM           13 ///< triSum
#define PRECISION        14 ///< precision
#define BESTOMEGA        15 ///< bestOmega
#define NUM_KERNELS      16 ///< overall number of kernels

namespace SFOP {

/**
 * @brief Class implementing CImage on the GPU using OpenCL
 * 
 * This implementation employs parallel computing using OpenCL.
 * Especially the 14 convolutions per layer for the SFOP scale space can be
 * computed very efficiently on the GPU.
 *
 * All image data remains stored on global GPU memory to reduce the amount of
 * data transfer between host and device.
 *
 * @see CImage
 */
class CImageCl : public CImage
{
    private:

        /// Global memory on GPU containing image data
        cl_mem m_mem_p;

        /// Image width
        unsigned int m_width;

        /// Image height
        unsigned int m_height;

        /// OpenCL controller
        static COpenCl* s_opencl_p;

    public:

        /**
         * @brief Return the width of the image
         *
         * @return Width
         */
        unsigned int width() const
        {
            return m_width;
        };

        /**
         * @brief Return the height of the image
         *
         * @return Height
         */
        unsigned int height() const
        {
            return m_height;
        };

        /**
         * @brief Return pointer to OpenCL memory
         *
         * @return OpenCL memory object
         */
        cl_mem mem_p() const
        {
            return m_mem_p;
        };

        /// Start up OpenCL
        void initOpenCl();

        /**
         * @brief Allocate global memory on OpenCL device
         *
         * @param[in] f_size Required memory size
         *
         * @return Pointer to OpenCL memory object
         */
        cl_mem newMem(
            const size_t f_size) const;

        /**
         * @brief Free global memory on OpenCL device
         *
         * @param[in] f_mem_p Pointer to memory object
         */
        void freeMem(
            cl_mem f_mem_p) const;

        /// Standard constructor
        CImageCl();

        /**
         * @brief Construct from OpenCL memory (in-place, no copy)
         *
         * @param[in] f_mem_p Memory on GPU
         * @param[in] f_width Image width
         * @param[in] f_height Image height
         */
        CImageCl(
            const cl_mem f_mem_p,
            const unsigned int f_width,
            const unsigned int f_height);

        /**
         * @brief Construct from CImg<float> image
         *
         * @param[in] f_img Image of class CImg<float>
         */
        CImageCl(
            const cimg_library::CImg<float> &f_img);

        /**
         * @brief Construct 1D row filter from filter name and scale
         *
         * @param[in] f_filterName Name of filter to create
         * @param[in] f_sigma Scale of Gaussian filter
         */
        CImageCl(
            const EFilterNames f_filterName,
            const float f_sigma);

        /// Standard destructor
        ~CImageCl();

        /**
         * @brief Load from file
         *
         * @param[in] f_filename[]
         */
        void load(
            const char f_filename[]);

        /**
         * @brief Execute OpenCL kernel
         *
         * @param[in] f_kernel           Operation index
         * @param[in] f_globalWorksize_p Pointer to global work size (default: 1)
         * @param[in] f_localWorksize_p  Pointer to local work size (default: 1)
         * @param[in] f_dim              Dimension (default: 1)
         */
        void runKernel(
            const unsigned int f_kernel,
            const size_t* f_globalWorksize_p = NULL,
            const size_t* f_localWorksize_p = NULL,
            const unsigned int f_dim = 1) const;

        /**
         * @brief Convolve image with given filters
         *
         * @param[in] f_rowFilter_p Convolution row filter
         * @param[in] f_colFilter_p Convolution column filter
         */
        CImage* conv(
            const CImage *f_rowFilter_p,
            const CImage *f_colFilter_p);

        /**
         * @brief Compute three squared gradients
         *
         * @param[in] f_gx_p   Gradient gx
         * @param[in] f_gy_p   Gradient gy
         * @param[out] f_gx2_p Squared gradient gx^2
         * @param[out] f_gxy_p Squared gradient gx * gy
         * @param[out] f_gy2_p Squared gradient gy^2
         */
        void triSqr(
                const CImage* f_gx_p,
                const CImage* f_gy_p,
                CImage* &f_gx2_p,
                CImage* &f_gxy_p,
                CImage* &f_gy2_p) const;

        /**
         * @brief Compute smaller eigenvalue of symmetrix matrix scaled by some factor M
         *
         * The scaled smaller eigenvalue lambda2 is defined as
         * @code
         *   lambda2 = M * (trace / 2 + sqrt(trace^2 / 4 - determinant))
         * @endcode
         * with trace and determinant being
         * @code
         *   trace = Nxx + Nyy
         *   determinant = Nxx * Nyy - Nxy^2;
         * @endcode
         *
         * @param[in] f_M     Scale factor M
         * @param[in] f_Nxx_p Upper left element
         * @param[in] f_Nxy_p Secondary diagonal element
         * @param[in] f_Nyy_p Lower right element
         *
         * @return Smaller eigenvalue scaled by M
         */
        CImage* lambda2(
                const float f_M,
                const CImage* f_Nxx_p,
                const CImage* f_Nxy_p,
                const CImage* f_Nyy_p) const;

        /**
         * @brief Compute three rotated, squared gradients
         *
         * The rotated gradients are defined as
         * @code
         *   gxa =   gx * cos(alpha) + gy * sin(alpha)
         *   gya = - gx * sin(alpha) + gy * cos(alpha)
         * @endcode
         *
         * @param[in]  f_alpha   Rotation angle
         * @param[in]  f_gx_p    Gradient gx
         * @param[in]  f_gy_p    Gradient gy
         * @param[out] f_gx2a_p  Square of first rotated gradient: gxa^2
         * @param[out] f_2gxya_p Twice the mixed product: 2 * gxa * gxy
         * @param[out] f_gy2a_p  Square of second rotated gradient: gya^2
         */
        void triSqrAlpha(
                const float f_alpha,
                const CImage* f_gx_p,
                const CImage* f_gy_p,
                CImage* &f_gx2a_p,
                CImage* &f_2gxya_p,
                CImage* &f_gy2a_p) const;

        /**
         * @brief Compute sum of three images
         *
         * @param[in] f_a_p First image
         * @param[in] f_b_p Second image
         * @param[in] f_c_p Third image
         *
         * @return Sum of all three images
         */
        CImage* triSum(
                const CImage* f_a_p,
                const CImage* f_b_p,
                const CImage* f_c_p) const;

        /**
         * @brief Compute precision from smaller eigenvalue, error sum and scaling factor
         *
         * The precision is defined as
         * @code
         *   precision = factor * lambda2 / omega
         * @endcode
         *
         * @param[in] f_factor    Scaling factor
         * @param[in] f_lambda2_p Smaller eigenvalue
         * @param[in] f_omega_p   Error sum
         *
         * @return  Precision
         */
        CImage* precision(
                const float f_factor,
                const CImage* f_lambda2_p,
                const CImage* f_omega_p) const;

        /**
         * @brief Compute best error sum from three error sums for a different * angle each
         *
         * The best error sum is defined as
         * @code
         *   omegaBest = a - b
         * @endcode
         * with
         * @code
         *   a = 1/3 * (omega0 + omega60 + omega120)
         *   b = 2/3 * sqrt(omega0^2 + omega60^2 + omega120^2 - omega0 * omega60 - omega60 * omega120 - omega120 * omega0)
         * @endcode
         *
         * @param[in] f_omega0_p   Error sum for 0 degrees angle
         * @param[in] f_omega60_p  Error sum for 60 degrees angle
         * @param[in] f_omega120_p Error sum for 120 degrees angle
         *
         * @return Best error sum
         */
        CImage* bestOmega(
                const CImage* f_omega0_p,
                const CImage* f_omega60_p,
                const CImage* f_omega120_p) const;

        /// Reduce size by factor 2. Sample nearest neighbor pixels. In-place
        CImage* downsample();

        /**
         * @brief Clones the image
         *
         * @return Cloned image
         */
        CImage* clone() const;

        /**
         * @brief Return image as CImg<float>
         *
         * @return Image as CImg<float>
         */
        cimg_library::CImg<float> asCimg() const;

        /**
         * @brief Identifies local maxima for three given layers.
         *
         * Identifies local maxima for three given images that are assumed to be
         * stacked over each other in an equal distance. The maxima are
         * calculated pixel-wise in a local neighborhood of size 3x3x3.
         *
         * @param[in,out] f_features_p Feature set to push back new features
         * @param[in] f_below_p Layer below
         * @param[in] f_above_p Layer above
         * @param[in] f_lambda2_p Image with lambda2 values of current layer
         * @param[in] f_sigma Integration scale sigma of current layer w.r.t. downsampled image
         * @param[in] f_numLayers Number of layers per octave
         * @param[in] f_factor Scale factor >= 1 w.r.t. original image size
         */
        virtual void findLocalMax(
            std::list<CFeature>* f_features_p,
            const CImage* f_below_p,
            const CImage* f_above_p,
            const CImage* f_lambda2_p,
            const float f_sigma,
            const unsigned int f_numLayers,
            const unsigned int f_factor) const;

        /**
         * @brief Compute gradient of a 3D image cube
         * @warning Very Slow! Only for test purposes.
         *
         * @param[in] f_cube CImg of size 3x3x3
         *
         * @return Gradient in form of 3x1 CImg (column vector)
         */
        cimg_library::CImg<float> computeGradient(
            const cimg_library::CImg<float> &f_cube) const;

        /**
         * @brief Compute Hessian of a 3x3x3 CImg matrix
         * @warning Very Slow! Only for test purposes.
         *
         * @param[in] f_cube CImg of size 3x3x3
         *
         * @return Hessian in form of 3x3 CImg
         */
        cimg_library::CImg<float> computeHessian(
            const cimg_library::CImg<float> &f_cube) const;

        /**
         * @brief Compute inverse of a matrix
         * @warning Very Slow! Only for test purposes.
         *
         * @param[in] f_matrix CImg matrix of size 3x3
         *
         * @return Inverse in form of 3x3 CImg
         */
        cimg_library::CImg<float> computeInverse(
            const cimg_library::CImg<float> &f_matrix) const;

        /**
         * @brief Solve linear system -H * x = g
         * @warning Very Slow! Only for test purposes.
         *
         * @param[in] f_matrix Matrix H in form of 3x3 CImg
         * @param[in] f_vector Vector g in form of 3x1 CImg (column vector)
         *
         * @return Solution x in form of 3x1 CImg (column vector)
         */
        cimg_library::CImg<float> computeSolver(
            const cimg_library::CImg<float> &f_matrix,
            const cimg_library::CImg<float> &f_vector) const;

        /**
         * @brief Check if a matrix is negative definite
         * @warning Very Slow! Only for test purposes.
         *
         * https://www.fachschaft5.de/forum/index.php?page=Thread&threadID=13564&postID=108217
         *
         * @param[in] f_matrix CImg matrix of size 3x3
         *
         * @return True if matrix is negative definite
         */
        bool computeIfNegDefinite(
            const cimg_library::CImg<float> &f_matrix) const;

};

};

#endif
