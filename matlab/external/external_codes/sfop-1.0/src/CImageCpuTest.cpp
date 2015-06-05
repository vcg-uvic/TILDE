#include "CImageCpuTest.h"

namespace SFOP {

CPPUNIT_TEST_SUITE_REGISTRATION (CImageCpuTest);

void CImageCpuTest::setUp(void)
{
    m_imageFactory_p = new CImageFactory<CImageCpu>();
    setUpVariables();
}

void CImageCpuTest::gradientTest()
{
    compareImages(m_cg1, CImageCpu::computeGradient(m_cP1));
}

void CImageCpuTest::hesseTest()
{
    compareImages(m_cH1, CImageCpu::computeHessian(m_cP1));
}

void CImageCpuTest::solverTest()
{
    cimg_library::CImg<float> l_cgradient(m_cg1);
    compareImages(m_cupdate1, -l_cgradient.solve(m_cH1), 1e-4);
}

void CImageCpuTest::negativeDefiniteTest()
{
    CPPUNIT_ASSERT_EQUAL(true,  CImageCpu::checkNegativeDefinite(m_cH1));
    CPPUNIT_ASSERT_EQUAL(false, CImageCpu::checkNegativeDefinite(m_cH2));
    CPPUNIT_ASSERT_EQUAL(false, CImageCpu::checkNegativeDefinite(m_cH3));
    CPPUNIT_ASSERT_EQUAL(true,  CImageCpu::checkNegativeDefinite(m_cH4));
}

}

