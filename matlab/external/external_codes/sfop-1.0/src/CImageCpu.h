#ifndef CIMAGECPUV2_H_
#define CIMAGECPUV2_H_

#include "CImg.h"
#include "CImage.h"
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

namespace SFOP {

/**
 * @brief Class implementing CImage on the CPU using the CImg library
 *
 * This implementation is for machines without OpenCl enabled GPU hardware and
 * acts as reference for the GPU implementation in terms of both accuray and
 * speed.
 *
 * @see CImage
 */
class CImageCpu : public CImage, private cimg_library::CImg<float>{

    public:

        /**
         * @brief Compute gradient of a 3D image cube
         *
         * @param[in] f_cube CImg of size 3x3x3
         *
         * @return Gradient in form of 3x1 CImg (column vector)
         */
        static cimg_library::CImg<float> computeGradient(
                const cimg_library::CImg<float> &f_cube);

        /**
         * @brief Compute Hessian of a 3x3x3 CImg matrix
         *
         * @param[in] f_cube CImg of size 3x3x3
         *
         * @return Hessian in form of 3x3 CImg
         */
        static cimg_library::CImg<float> computeHessian(
                const cimg_library::CImg<float> &f_cube);

        /**
         * @brief Check if a matrix is negative definite
         *
         * https://www.fachschaft5.de/forum/index.php?page=Thread&threadID=13564&postID=108217
         * @param[in] f_m CImg matrix of size 3x3
         *
         * @return True if matrix is negative definite
         */
        static bool checkNegativeDefinite(
                const cimg_library::CImg<float> &f_m);

    public:

        /**
         * @brief Return the width of the image
         *
         * @return Width
         */
        unsigned int width() const { return cimg_library::CImg<float>::width(); };

        /**
         * @brief Return the height of the image
         *
         * @return Height
         */
        unsigned int height() const { return cimg_library::CImg<float>::height(); };

        /// Standard constructor
        CImageCpu() { };

        /**
         * @brief Construct from CImg<float> image
         *
         * @note Currently only used for unit tests.
         *
         * @param[in] f_img Image of class CImg<float>
         */
        CImageCpu(
                const cimg_library::CImg<float> &f_img) : cimg_library::CImg<float>(f_img) { };

        /**
         * @brief Construct 1D row filter from filter name and scale
         *
         * @param[in] f_filterName Name of filter to create
         * @param[in] f_sigma Scale of Gaussian filter
         */
        CImageCpu(
                const EFilterNames f_filterName,
                const float f_sigma);

        /// Destructor
        ~CImageCpu() { };

        /**
         * @brief Load from file
         *
         * @param[in] f_filename[]
         */
        void load(
                const char f_filename[]);

        /**
         * @brief Convolve image with given filters. In-place
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
        virtual CImage* clone() const;

        /**
         * @brief Return image as CImg<float>
         *
         * @return Image as CImg<float>
         */
        cimg_library::CImg<float> asCimg() const;

        /**
         * @brief Identifies local maxima for three given layers.
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

};

};

#endif

