#ifndef CSFOP_H 
#define CSFOP_H

#include "CImage.h"
#include "COctave.h"
#include "CImageFactory.h"
#include <fstream>
#include <math.h>
#include <float.h>

namespace SFOP {

/**
 * @mainpage C++ Implementation of the SFOP Keypoint Detector
 *
 * @author
 *   Theory: Wolfgang Foerstner, Timo Dickscheid and Falko Schindler \n
 *   Implementation: Falko Schindler with support by Daria Makarova, Magnus Becker and Thomas Läbe\n
 *   Department of Photogrammetry, University of Bonn, Germany
 *
 * @date April 2012
 *
 * @note
 * For further information refere to the
 * <A HREF="http://www.ipb.uni-bonn.de/sfop">SFOP project website</A>.\n
 * SFOP has been
 * <A HREF="http://www.ipb.uni-bonn.de/uploads/tx_ikgpublication/foerstner09.detecting.pdf">
 * published at ICCV'09.</A>\n
 * Please cite as:
 * \verbatim
@inproceedings{
    foerstner*09:detecting,
    author={W. F\"orstner and T. Dickscheid and F. Schindler},
    title={Detecting Interpretable and Accurate Scale-Invariant Keypoints},
    booktitle={12th IEEE International Conference on Computer Vision (ICCV'09)},
    address={Kyoto, Japan},
    year={2009}
} \endverbatim
 * No warranty for validity of this implementation.\n
 * SFOP is distributed under the <A
 * HREF="http://www.gnu.org/licenses/lgpl.html">GNU Lesser General Public License</A>.
 */

/**
 * @brief Class implementing the SFOP detector.
 *
 * The constructor loads an image file.
 * The function detect() builds the SFOP scale space and detects local maxima.
 * Furthermore it filters features w.r.t. their precision and the smaller
 * eigenvalue of the structure tensor.
 * Given a value for the expected image noise this filtering step is realized in
 * a statistically motivated manner.
 * During a non-maxima suppression step dense feature cluster will be reduced to
 * the most precise feature.
 */
class CSfop
{
    private:

        /// Image where to detect SFOP features.
        CImage *m_image_p;

        /// Set of detected features
        std::list<CFeature>* m_features_p;

        /// Creates CImages of the correct type.
        const CImageFactoryAbstract* m_factory_p;

        /**
         * @brief Suppresses less precise features if very close to each other
         *
         * @param[in] f_numLayers Number of layers per octave
         */
        void nonMaximaSuppression(
                const unsigned int f_numLayers);

    public:
        /// @warning: Should be private. Is public for unit testing only.

        /**
         * @brief Compute the inverse chi-square distribution
         *
         * @param[in] f_p Probability
         * @param[in] f_v Degrees of freedom
         *
         * @return Inverse of the chi-square cumulative distribution function
         */
        static double chi2inv(
                const double f_p,
                const double f_v);

        /**
         * @brief Natural logarithm of the gamma function
         *
         * @param[in] f_x Input value
         *
         * @return Logarithm of the corresponding value of the gamma function
         */
        static double gammaln(
                const double f_x);

        /**
         * @brief Incomplete gamma function
         *
         * @param[in] f_x Input parameter X
         * @param[in] f_a Input parameter A
         *
         * @return Result is defined as 1 / gamma(a) * integral from 0 to x of t^(a-1) exp(-t) dt
         */
        static double gammainc(
                const double f_x,
                const double f_a);

        /**
         * @brief Core algorithm for error functions
         *
         * @param[in] f_x Input parameter X
         *
         * @return Error function for JINT = 4
         */
        static double erfcore(
                const double f_x);

        /**
         * @brief Removes unprecise, elongated and boundary features according to two thresholds
         *
         * @param[in] f_imageNoise Image noise 
         * @param[in] f_lambdaWeight Weight for eigenvalue lambda2
         * @param[in] f_precisionThreshold Precision threshold
         * @param[in] f_width Original image width
         * @param[in] f_height Original image height
         */
        void filterFeatures(
                const float f_imageNoise,
                const float f_lambdaWeight,
                const float f_precisionThreshold,
                const unsigned int f_width,
                const unsigned int f_height);

    public:

        /** 
         * @brief Constructor. Load CImage from filename.
         * 
         * @param[in] f_factory_p Factory for creating CImage objects.
         * @param[in] f_filename[] Filename of input image
         */
        CSfop(
                const CImageFactoryAbstract* f_factory_p,
                const char f_filename[]);

        /// Destructor
        ~CSfop();

        /** 
         * @brief Detector. Create CFeatureSet from CImage.
         * 
         * @param[in] f_numOctaves         Number of octaves
         * @param[in] f_numScalesPerOctave Number of scales per octave
         * @param[in] f_precisionThreshold Precision threshold for features
         * @param[in] f_imageNoise         Value for the amount of noise in the image
         * @param[in] f_lambdaWeight       Influences how strong lambda is weighted during filtering
         * @param[in] f_type               Type of features to be detected (positive angle in degrees or -1 for "optimal")
         */
        void detect(
                const unsigned int f_numOctaves,
                const unsigned int f_numScalesPerOctave,
                const float f_precisionThreshold,
                const float f_imageNoise,
                const float f_lambdaWeight,
                const float f_type);

        /** 
         * @brief Write detected features to given file.
         * 
         * @param[in] f_filename[] Name of keypoint file
         */
        void writeFile(const char f_filename[]) const;

        /**
         * @brief Pointer to feature list
         *
         * @return Pointer to feature list
         */
        std::list<CFeature>* features_p() { return m_features_p; };

};

};

#endif
