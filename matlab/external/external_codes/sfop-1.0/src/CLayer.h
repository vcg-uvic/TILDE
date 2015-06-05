#ifndef CLAYER_H
#define CLAYER_H

#include "CImage.h"
#include "CImageFactory.h"
#include "CFeature.h"
#include <list>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

namespace SFOP {

/**
 * @brief One layer of the SFOP scale space.
 *
 * There are two steps for detecting features in one layer:
 * -# Construct a layer of the SFOP scale space from the input image, layer and
 *  octave indices.
 * -# Detect local maxima given the upper and lower neighboring layer.
 */
class CLayer
{
    private:

        /// Holds the factory used for creating the correct type of images.
        const CImageFactoryAbstract* m_factory_p;

        /// Index of this layer within its octave
        const int m_layer;
            
        /// Index of this layer's octave within image pyramid
        const unsigned int m_octave;
        
        /// Number of layers per octave
        const unsigned int m_numLayers;

        /// Precision array
        CImage* m_precision_p;

        /// Array of smaller eigenvalues
        CImage* m_lambda2_p;

    public:

        /** 
         * @brief Constructor. Compute smaller eigenvalues and precision
         * 
         * @param[in] f_factory   Factory for creating CImage objects
         * @param[in] f_img_p     Pointer to image
         * @param[in] f_layer     Index of this layer within octave
         * @param[in] f_octave    Index of this layer's octave
         * @param[in] f_numLayers Number of layers within this layer's octave
         * @param[in] f_type      Type of features to be detected (positive angle in degrees or -1 for "optimal")
         */
        CLayer(
                const CImageFactoryAbstract *f_factory,
                const CImage *f_img_p,
                const int f_layer,
                const unsigned int f_octave,
                const unsigned int f_numLayers,
                const float f_type);

        /// Destructor
        ~CLayer();

        /** 
         * @brief Detect features in this layer. Find maxima in 3D scale space
         * and optimize pixel position like Lowe2004
         * 
         * @param[in,out] f_features_p Feature set to push back new features
         * @param[in] f_below_p CLayer located below in scale (smaller sigma)
         * @param[in] f_above_p CLayer located above in scale (larger sigma)
         */
        void detect(
                std::list<CFeature>* f_features_p,
                const CLayer* f_below_p,
                const CLayer* f_above_p) const;

        /**
         * @brief Pointer to eigenvalue array
         *
         * @return Pointer to eigenvalue array
         */
        CImage* lambda2_p() const { return m_lambda2_p; };

        /**
         * @brief Pointer to precision array
         *
         * @return Pointer to precision array
         */
        CImage* precision_p() const { return m_precision_p; };

};

};

#endif
