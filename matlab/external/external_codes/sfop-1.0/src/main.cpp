#include "CSfop.h"
#include "CImg.h"
#include "CImageFactory.h"
#include "CImageCpu.h"
#ifdef GPU
#include "CImageCl.h"
#endif

using namespace cimg_library;
#undef min
#undef max
using namespace SFOP;

/**
 * @brief This program demonstrates the use of the SFOP detector class CSfop.
 *
 * Run \code ./main --help \endcode to see a list of possible arguments.
 *
 * @param[in] argc
 * @param[in] argv[]
 *
 * @return 0 if succeeded
 */
int main(int argc, char* argv[])
{
    // parse options
    const char* inFile = cimg_option("-i", "../examples/lena.png", "Input image");
    const char* outFile = cimg_option("-o", "", "Output keypoint file");
    const bool display = cimg_option("--display", true, "Display result");
    const float imageNoise = cimg_option("--noise", 0.02, "Image noise");
    const float precisionThreshold = cimg_option("--pTresh", 0, "Threshold on precision");
    const float lambdaWeight = cimg_option("--lWeight", 2, "Weighting for lambda");
    const unsigned int numOctaves = cimg_option("--numOctaves", 3, "Number of octaves");
    const unsigned int numLayers = cimg_option("--numLayers", 4, "Number of layers per octave");
#ifdef GPU
    const int device = cimg_option("--device", 0, "Device, 0 for CPU, 1 for OpenCL on GPU");
#endif
    const int type = cimg_option("--type", -1, "Angle in degrees, or -1 for optimal features");

    // print help text
    if (argc < 2) printf("Try `%s --help` for a list of possible arguments.\n", argv[0]);
    if (argc < 3) return 0;

    // create image factory depending on device
    CImageFactoryAbstract* l_factory_p;
#ifdef GPU
    if (device == 0) {
        l_factory_p = new CImageFactory<CImageCpu>();
    }
    else {
        l_factory_p = new CImageFactory<CImageCl>();
    }
#else
    l_factory_p = new CImageFactory<CImageCpu>();
#endif

    // process input image
    CSfop detector(l_factory_p, inFile);
    detector.detect(numOctaves, numLayers, imageNoise, lambdaWeight, precisionThreshold, type);
    if (outFile[0] != 0) {
        detector.writeFile(outFile);
    }
    if (display) {
        CImageCpu image;
        image.load(inFile);
        image.displayFeatures(detector.features_p());
    }

    return 0;
}
