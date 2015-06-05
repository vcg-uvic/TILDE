#include "CImageClTest.h"

namespace SFOP {

CPPUNIT_TEST_SUITE_REGISTRATION (CImageClTest);

void CImageClTest::setUp(void)
{
    m_imageFactory_p = new CImageFactory<CImageCl>();
    setUpVariables();
}

void CImageClTest::gradientTest()
{
    CImageCl* l_test_p = new CImageCl(m_cg1);
    compareImages(m_cg1, l_test_p->computeGradient(m_cP1));
    delete l_test_p;
    l_test_p = NULL;
}

void CImageClTest::hesseTest()
{
    CImageCl* l_test_p = new CImageCl(m_cg1);
    compareImages(m_cH1, l_test_p->computeHessian(m_cP1));
    delete l_test_p;
    l_test_p = NULL;
}

void CImageClTest::inverseTest()
{
    CImageCl* l_test_p = new CImageCl(m_cg1);
    compareImages(m_cH1.get_invert(), l_test_p->computeInverse(m_cH1), 1e-3);
    delete l_test_p;
    l_test_p = NULL;
}

void CImageClTest::solverTest()
{
    CImageCl* l_test_p = new CImageCl(m_cg1);
    compareImages(m_cupdate1, l_test_p->computeSolver(m_cH1, m_cg1), 1e-4);

    cimg_library::CImg<float> l_P27(3, 3, 3, 1, 0.0f);
    l_P27(1, 1, 1) = 1.0f;
    l_P27(2, 1, 1) = 0.1f;
    cimg_library::CImg<float> l_H = l_test_p->computeHessian(l_P27);
    cimg_library::CImg<float> l_g = l_test_p->computeGradient(l_P27);
    cimg_library::CImg<float> l_update = l_test_p->computeSolver(l_H, l_g);
    compareImages(cimg_library::CImg<float>(1, 3, 1, 1, 0.026316f, 0.0f, 0.0f), l_update);
    delete l_test_p;
    l_test_p = NULL;
}

void CImageClTest::negativeDefiniteTest()
{
    CImageCl* l_test_p = new CImageCl(m_cg1);
    CPPUNIT_ASSERT_EQUAL(true,  l_test_p->computeIfNegDefinite(m_cH1));
    CPPUNIT_ASSERT_EQUAL(false, l_test_p->computeIfNegDefinite(m_cH2));
    CPPUNIT_ASSERT_EQUAL(false, l_test_p->computeIfNegDefinite(m_cH3));
    CPPUNIT_ASSERT_EQUAL(true,  l_test_p->computeIfNegDefinite(m_cH4));
    delete l_test_p;
    l_test_p = NULL;
}

}

