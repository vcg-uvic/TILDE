#ifndef CIMAGE_H 
#define CIMAGE_H

#include "CImg.h"
#include "CFeature.h"
#include <list>

namespace SFOP {

/**
 * @brief Abstract class declaring an image
 *
 * All basis operations for building the SFOP scale space and detecting features
 * are declared.
 * The implementation, i.e. computations and memory management, are be done on
 * CPU or GPU in derived classes like CImageCpu and CImageCl.
 *
 * Building the scale space is devided into several intermediate steps
 * intercepted by convolutions, i.e. integrating over some neighboring pixels.
 * Detecting local maxima in one layer of the scale space is done the one
 * function findLocalMax().
 *
 * For reading image files or displaying results the library CImg is used.
 * Therefore the class declaration of CImage is closely related to CImg<float>.
 */
class CImage
{

    public:

        /// Possible filter names
        enum EFilterNames {
            G,  ///< Gaussian kernel
            Gx, ///< Derivative of Gaussian kernel
            xG, ///< Gaussian kernel multiplied with x
            x2G ///< Gaussian kernel multiplied with x^2
        };

        /// Destructor
        virtual ~CImage() { };

        /** 
         * @brief Return the width of the image
         * 
         * @return Width
         */
        virtual unsigned int width() const = 0;

        /** 
         * @brief Return the height of the image
         * 
         * @return Height
         */
        virtual unsigned int height() const = 0;

        /** 
         * @brief Load from file
         * 
         * @param[in] f_filename[]
         */
        virtual void load(
                const char f_filename[]) = 0;

        /** 
         * @brief Convolve image with given filters. In-place
         * 
         * @param[in] f_rowFilter_p Convolution row filter
         * @param[in] f_colFilter_p Convolution column filter
         */
        virtual CImage* conv(
                const CImage *f_rowFilter_p,
                const CImage *f_colFilter_p) = 0;

        /**
         * @brief Compute three squared gradients
         *
         * @param[in] f_gx_p   Gradient gx
         * @param[in] f_gy_p   Gradient gy
         * @param[out] f_gx2_p Squared gradient gx^2
         * @param[out] f_gxy_p Squared gradient gx * gy
         * @param[out] f_gy2_p Squared gradient gy^2
         */
        virtual void triSqr(
                const CImage* f_gx_p,
                const CImage* f_gy_p,
                CImage* &f_gx2_p,
                CImage* &f_gxy_p,
                CImage* &f_gy2_p) const = 0;

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
        virtual CImage* lambda2(
                const float f_M,
                const CImage* f_Nxx_p,
                const CImage* f_Nxy_p,
                const CImage* f_Nyy_p) const = 0;

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
        virtual void triSqrAlpha(
                const float f_alpha,
                const CImage* f_gx_p,
                const CImage* f_gy_p,
                CImage* &f_gx2a_p,
                CImage* &f_2gxya_p,
                CImage* &f_gy2a_p) const = 0;

        /**
         * @brief Compute sum of three images
         *
         * @param[in] f_a_p First image
         * @param[in] f_b_p Second image
         * @param[in] f_c_p Third image
         *
         * @return Sum of all three images
         */
        virtual CImage* triSum(
                const CImage* f_a_p,
                const CImage* f_b_p,
                const CImage* f_c_p) const = 0;

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
        virtual CImage* precision(
                const float f_factor,
                const CImage* f_lambda2_p,
                const CImage* f_omega_p) const = 0;

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
        virtual CImage* bestOmega(
                const CImage* f_omega0_p,
                const CImage* f_omega60_p,
                const CImage* f_omega120_p) const = 0;

        /// Reduce size by factor 2. Sample nearest neighbor pixels. In-place
        virtual CImage* downsample() = 0;
        
        /**
         * @brief Clones the image
         *
         * @return Cloned image
         */
        virtual CImage* clone() const = 0;

        /// Display image
        virtual void display() const { this->asCimg().display(); };

        /** 
         * @brief Display image with features
         *
         * @param[in] f_features_p List of with features to be shown
         */
        virtual void displayFeatures(
                const std::list<CFeature>* f_features_p) const
        {
            cimg_library::CImg<float> l_img = this->asCimg();
            float l_white[1] = { 1.0f };
            float l_black[1] = { 0.0f };
            for (std::list<CFeature>::const_iterator
                    l_iter = f_features_p->begin();
                    l_iter != f_features_p->end(); l_iter++) {
                l_img.draw_circle(l_iter->x, l_iter->y, l_iter->sigma + 1, l_black, 1.0f, 1);
                l_img.draw_circle(l_iter->x, l_iter->y, l_iter->sigma    , l_white, 1.0f, 1);
                l_img.draw_circle(l_iter->x, l_iter->y, l_iter->sigma - 1, l_black, 1.0f, 1);
            }
            l_img.display();
        };

        /**
         * @brief Return image as CImg<float>
         * 
         * @return Image as CImg<float>
         */
        virtual cimg_library::CImg<float> asCimg() const = 0;

        /**
         * @brief Identifies local maxima for three given layers.
         *
         * Identifies local maxima for three given images that are assumed to be
         * stacked over each other in an equal distance. The maxima are
         * calculated pixel-wise in a local neighborhood of size 3x3x3.
         *
         * @param[in,out] f_features_p List of features
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
                const unsigned int f_factor) const = 0;

};

};

#endif
