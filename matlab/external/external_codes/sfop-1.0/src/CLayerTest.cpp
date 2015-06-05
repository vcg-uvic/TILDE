#include "CLayerTest.h"

namespace SFOP {

CPPUNIT_TEST_SUITE_REGISTRATION (CLayerTest);

void CLayerTest::constructorTest()
{
    CImageFactoryAbstract* l_imageFactory_p;
    for (int i = 0; i <= 1; ++i) {

        // generate image factory
#ifdef GPU
        if (i == 0) {
            l_imageFactory_p = new CImageFactory<CImageCpu>();
        }
        else {
            l_imageFactory_p = new CImageFactory<CImageCl>();
        }
#else
        l_imageFactory_p = new CImageFactory<CImageCpu>();
#endif

        // load input image
        CImage* l_image_p = l_imageFactory_p->createImage();
        l_image_p->load("../examples/dot.png");

        // create layer
        CLayer l_layer(l_imageFactory_p, l_image_p, 1, 2, 4, -1.0f);

        // check lambda2
        cimg_library::CImg<float> l_lambda2 = l_layer.lambda2_p()->asCimg().crop(0, 0, 127, 127);
        CPPUNIT_ASSERT_EQUAL(128, l_lambda2.width());
        CPPUNIT_ASSERT_EQUAL(128, l_lambda2.height());
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.000057,      l_lambda2.min(),       1e-5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.411539,      l_lambda2.max(),       1e-1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.005936,      l_lambda2.mean(),      1e-2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.032385, sqrt(l_lambda2.variance()), 1e-2);

        // check precision
        cimg_library::CImg<float> l_precision = l_layer.precision_p()->asCimg().crop(0, 0, 127, 127);
        CPPUNIT_ASSERT_EQUAL(128, l_precision.width());
        CPPUNIT_ASSERT_EQUAL(128, l_precision.height());
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.107608,      l_precision.min(),       1e-1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(43.790334,      l_precision.max(),       1e+1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 5.055385,      l_precision.mean(),      1e-0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( 2.348089, sqrt(l_precision.variance()), 1e-0);

        // delete image and layer
        delete l_image_p;
        l_image_p = NULL;

    }

}

}

