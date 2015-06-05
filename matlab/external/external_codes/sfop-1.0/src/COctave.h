#ifndef COCTAVE_H
#define COCTAVE_H

#include "CImage.h"
#include "CLayer.h"
#include "CImageFactory.h"
#include "CFeature.h"
#include <list>
#include <vector>
#include <math.h>

namespace SFOP {

/**
 * @brief Class for holding multiple layers of one octave.
 *
 * The SFOP scale space is devided into octaves, each having the same number of
 * layers and being computed from a downsampled version of the original image.
 * Every octave has have the resolution of its predecessor.
 */
class COctave
{

    private:

        /// Image factory
        const CImageFactoryAbstract* m_factory_p;

        /// Pre-scaled image to work on (scaled to first layer of octave)
        const CImage* m_image_p;

        /// Number of layers in this octave
        const unsigned int m_numLayers;

        /// Index of this octave, starting with 0 being the original image
        const unsigned int m_octave;

        /// Type of features to be detected (positive angle in degrees or -1 for "optimal")
        const float m_type;

        /// Vector containing the layers of the current octave
        std::vector<CLayer*> m_layers;

        /// Set of detected features
        std::list<CFeature> m_features;

    public:

        /** 
         * @brief Constructor
         * 
         * @param[in] f_factory_p Factory for creating CImage objects
         * @param[in] f_image_p   Pre-scaled image to work on (scaled to first layer of octave)
         * @param[in] f_octave    Index of this octave
         * @param[in] f_numLayers Number of scales in this octave
         * @param[in] f_type      Type of features to be detected (positive angle in degrees or -1 for "optimal")
         */
        COctave(
            const CImageFactoryAbstract* f_factory_p,
            const CImage* f_image_p,
            const int f_octave,
            const unsigned int f_numLayers,
            const float f_type) : m_factory_p(f_factory_p),
            m_image_p(f_image_p), m_numLayers(f_numLayers), m_octave(f_octave), m_type(f_type) { };

        /// Destructor
        ~COctave() { };

        /**
         * @brief Detect features within this octave
         *
         * @param[in,out] f_features_p List of features
         */
        void detect(
            std::list<CFeature>* f_features_p);

};

};

#endif
