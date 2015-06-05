#ifndef CFEATURE_H
#define CFEATURE_H

namespace SFOP {

/**
 * @brief Class defining one SFOP feature
 *
 * It contains the location (x,y), the scale sigma as well as precision and the
 * smaller eigenvalue of the structure tensor.
 */
class CFeature
{
    public:

        /// x-coordinate (column)
        float x;

        /// y-coordinate (row)
        float y;

        /// Size, i.e. integration scale with largest response
        float sigma;

        /// Smaller eigenvalue of the structure tensor
        float lambda2;

        /// Precision
        float precision;

        /** 
         * @brief Constructor
         * 
         * @param[in] f_x         x-coordinate (column)
         * @param[in] f_y         y-coordinate (row)
         * @param[in] f_sigma     Size
         * @param[in] f_lambda2   Eigenvalue lambda2
         * @param[in] f_precision Precision
         */
        CFeature(
                const float f_x,
                const float f_y,
                const float f_sigma,
                const float f_lambda2,
                const float f_precision) :
            x(f_x), y(f_y), sigma(f_sigma),
            lambda2(f_lambda2), precision(f_precision) { };

        /// Destructor
        ~CFeature() { };

};

};

#endif
