#ifndef CIMAGEFACTORY_H_
#define CIMAGEFACTORY_H_

#include "CImage.h"

namespace SFOP {

/**
 * @brief Class implementing an abstract factory for creating instances of CImage
 *
 * The factory is implemented in the derived template class CImageFactory.
 */
class CImageFactoryAbstract
{
    public:

        /**
         * @brief Create image
         *
         * @return Image pointer
         */
        virtual CImage* createImage() const = 0;

        /**
         * @brief Create from CImag<float>
         *
         * @param[in] f_cimg CImg image
         *
         * @return  Image pointer
         */
        virtual CImage* createImage(
                const cimg_library::CImg<float>& f_cimg) const = 0;

        /**
         * @brief Create Gaussian filter
         *
         * @param[in] f_filterName Filter name
         * @param[in] f_sigma Sigma of Gauss function
         *
         * @return Image pointer
         */
        virtual CImage* createImage(
                const CImage::EFilterNames f_filterName,
                const float f_sigma) const = 0;

};

/**
 * @brief Class implementing a templated specific factory for creating instances of CImage
 *
 * An arbitrary number of factories can be created, e.g. one for each device or specialization.
 *
 * @warning There may arise incompatibilities when the same device is used for several factories.
 * CImageCpu is fine, but CImageCL causes problems when using different OpenCL kernels concurrently on the same device.
 */
template <typename T> class CImageFactory : public CImageFactoryAbstract
{
    public:

        /// Standard constructor
        CImageFactory() { };

        /// Standard destructor
        virtual ~CImageFactory() { };

        /**
         * @brief Creates an image with the type dependent on the type of the factory
         *
         * @return Image pointer
         */
        virtual CImage* createImage() const { return new T(); };

        /**
         * @brief Creates an image from a CImg with the type dependent on the type of the factory
         *
         * @note Currently only used for unit tests
         *
         * @param[in] f_cimg CImg image
         *
         * @return Image pointer
         */
        virtual CImage* createImage(
                const cimg_library::CImg<float>& f_cimg) const { return new T(f_cimg); };

        /**
         * @brief Creates a filter with the image type dependent on the type of the factory
         *
         * @see CImage::EFilterNames for supported filter names
         *
         * @param[in] f_filterName Filter name
         * @param[in] f_sigma Sigma value
         *
         * @return Image pointer
         */
        virtual CImage* createImage(
                const CImage::EFilterNames f_filterName,
                const float f_sigma) const { return new T(f_filterName, f_sigma); };

};

};

#endif
