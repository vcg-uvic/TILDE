#include "CImageTest.h"

namespace SFOP {

void CImageTest::setUpVariables()
{
    const float l_P1[] = {
         0.512908935785717,  0.460483750542851,  0.350395373697577,
         0.095045735172483,  0.433671043922040,  0.709235197729075,
         0.115968232562751,  0.078084682150783,  0.369252908215510,
         0.033628378079100,  0.192150375429611,  0.471359845470339,
         0.144922819987369,  0.717835527713334,  0.661714278003999,
         0.431870413213908,  0.446034886150637,  0.508331533758124,
         0.528087872792076,  0.572878016425706,  0.360822066832717,
         0.336477257468029,  0.173266265156337,  0.086118482559043,
         0.393336369839188,  0.804367887230761,  0.011080687405113
    };
    const float l_g1[] = {
         0.105139437762620,  0.022549095108429, -0.005713531633706
    };
    const float l_H1[] = {
        -0.208891790690316, -0.032608270776741, -0.148086817305575,
        -0.032608270776741, -0.120883293040508,  0.070085510294924,
        -0.148086817305575,  0.070085510294924, -0.191085643689654
    };
    const float l_update1[] = {
         6.164381480886346, -5.414844777821789, -6.793180796592337
    };
    // Negative definite.

    const float l_H2[] = {
        -0.292783581380632, -0.315216541553483, -0.171173634611151,
        -0.315216541553483,  0.133233413918984,  0.140171020589849,
        -0.171173634611151,  0.140171020589849, -0.007171287379307
    };
    // Not negative definite.

    const float l_H3[] = {
        -0.001675372070947,  0.027175187669776,  0.055739548083273,
         0.027175187669776,  0.012350120878476, -0.039743469115227,
         0.055739548083273, -0.039743469115227, -0.198256931068961
    };
    // Not negative definite.

    const float l_H4[] = {
        -0.460567162761263, -0.255433083106965, -0.092347269222302,
        -0.255433083106965, -0.358533172162032,  0.030342041179697,
        -0.092347269222302,  0.030342041179697, -0.389342574758615
    };
    // Negative definite.

    m_cP1      = cimg_library::CImg<float>(l_P1,      3, 3, 3);
    m_cg1      = cimg_library::CImg<float>(l_g1,      1, 3);
    m_cupdate1 = cimg_library::CImg<float>(l_update1, 1, 3);
    m_cH1      = cimg_library::CImg<float>(l_H1,      3, 3);
    m_cH2      = cimg_library::CImg<float>(l_H2,      3, 3);
    m_cH3      = cimg_library::CImg<float>(l_H3,      3, 3);
    m_cH4      = cimg_library::CImg<float>(l_H4,      3, 3);
}

void CImageTest::compareImages(
        const cimg_library::CImg<float> &f_expected,
        const cimg_library::CImg<float> &f_result,
        const float f_eps) const
{
    CPPUNIT_ASSERT_EQUAL(f_expected.width(),  f_result.width());
    CPPUNIT_ASSERT_EQUAL(f_expected.height(), f_result.height());
    CPPUNIT_ASSERT_EQUAL(f_expected.depth(),  f_result.depth());
    const cimg_library::CImg<float> diff = f_expected - f_result;
    cimg_foroff(f_expected, i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(f_expected(i), f_result(i), f_eps);
    }
}

void CImageTest::filterTest()
{
    float l_filterG[] = {
         0.000044611342773,  0.000160094765892,  0.000514107605822,  0.001477324778400,  0.003798769940305,
         0.008740878047691,  0.017997500190861,  0.033159988420867,  0.054671578245654,  0.080659199898787,
         0.106485694027245,  0.125797983460311,  0.132984538550784,  0.125797983460311,  0.106485694027245,
         0.080659199898787,  0.054671578245654,  0.033159988420867,  0.017997500190861,  0.008740878047691,
         0.003798769940305,  0.001477324778400,  0.000514107605822,  0.000160094765892,  0.000044611342773};
    float l_filterGx[] = {
         0.000080047382946,  0.000234748131525,  0.000658615006254,  0.001642331167242,  0.003631776634645,
         0.007099365125278,  0.012209555186588,  0.018337039027397,  0.023749605738960,  0.025907057890795,
         0.022569391780762,  0.013249422261770,  0.000000000000000, -0.013249422261770, -0.022569391780762,
        -0.025907057890795, -0.023749605738960, -0.018337039027397, -0.012209555186588, -0.007099365125278,
        -0.003631776634645, -0.001642331167242, -0.000658615006254, -0.000234748131525, -0.000080047382946};
    float l_filterxG[] = {
        -0.000535336113272, -0.001761042424812, -0.005141076058220, -0.013295923005604, -0.030390159522444,
        -0.061186146333839, -0.107985001145163, -0.165799942104337, -0.218686312982616, -0.241977599696361,
        -0.212971388054489, -0.125797983460311,                  0,  0.125797983460311,  0.212971388054489,
         0.241977599696361,  0.218686312982616,  0.165799942104337,  0.107985001145163,  0.061186146333839,
         0.030390159522444,  0.013295923005604,  0.005141076058220,  0.001761042424812,  0.000535336113272};
    float l_filterx2G[] = {
         0.006424033359261,  0.019371466672929,  0.051410760582197,  0.119663307050433,  0.243121276179551,
         0.428303024336873,  0.647910006870980,  0.828999710521686,  0.874745251930465,  0.725932799089083,
         0.425942776108979,  0.125797983460311,                  0,  0.125797983460311,  0.425942776108979,
         0.725932799089083,  0.874745251930465,  0.828999710521686,  0.647910006870980,  0.428303024336873,
         0.243121276179551,  0.119663307050433,  0.051410760582197,  0.019371466672929,  0.006424033359261};
    CImage* l_filterG_p   = m_imageFactory_p->createImage(CImage::G, 3);
    CImage* l_filterGx_p  = m_imageFactory_p->createImage(CImage::Gx, 3);
    CImage* l_filterxG_p  = m_imageFactory_p->createImage(CImage::xG, 3);
    CImage* l_filterx2G_p = m_imageFactory_p->createImage(CImage::x2G, 3);
    compareImages(cimg_library::CImg<float>(l_filterG,   25, 1), l_filterG_p->asCimg(),   1e-4);
    compareImages(cimg_library::CImg<float>(l_filterGx,  25, 1), l_filterGx_p->asCimg(),  1e-4);
    compareImages(cimg_library::CImg<float>(l_filterxG,  25, 1), l_filterxG_p->asCimg(),  1e-4);
    compareImages(cimg_library::CImg<float>(l_filterx2G, 25, 1), l_filterx2G_p->asCimg(), 1e-4);
    delete l_filterG_p;
    delete l_filterGx_p;
    delete l_filterxG_p;
    delete l_filterx2G_p;
    l_filterG_p = NULL;
    l_filterGx_p = NULL;
    l_filterxG_p = NULL;
    l_filterx2G_p = NULL;
}

void CImageTest::widthTest()
{
    cimg_library::CImg<float> l_cimg(10, 5);
    l_cimg.rand(0.0f, 1.0f);
    const CImage* const l_img_p = m_imageFactory_p->createImage(l_cimg);
    CPPUNIT_ASSERT_EQUAL((int) l_cimg.width(), (int) l_img_p->width());
    delete l_img_p;
}

void CImageTest::heightTest()
{
    cimg_library::CImg<float> l_cimg(10, 5);
    l_cimg.rand(0.0f, 1.0f);
    const CImage* const l_img_p = m_imageFactory_p->createImage(l_cimg);
    CPPUNIT_ASSERT_EQUAL((int) l_cimg.height(), (int) l_img_p->height());
    delete l_img_p;
}

void CImageTest::convTest()
{
    const unsigned int l_radius = 7;
    cimg_library::CImg<float> l_cimg(1024, 1024);
    cimg_library::CImg<float> l_cfilterX(2 * l_radius + 1, 1);
    cimg_library::CImg<float> l_cfilterY(1, 2 * l_radius + 1);
    l_cimg.rand(0.0f, 1.0f);
    l_cfilterX.rand(0.0f, 0.1f);
    l_cfilterY.rand(0.0f, 0.1f);
    CImage* l_img_p = m_imageFactory_p->createImage(l_cimg);
    const CImage* const l_filterX_p = m_imageFactory_p->createImage(l_cfilterX);
    const CImage* const l_filterY_p = m_imageFactory_p->createImage(l_cfilterY);
    compareImages(
            l_cimg.convolve(l_cfilterX).convolve(l_cfilterY), 
            l_img_p->conv(l_filterX_p, l_filterY_p)->asCimg());
    delete l_img_p;
    delete l_filterX_p;
    delete l_filterY_p;
}

void CImageTest::triSqrTest()
{
    cimg_library::CImg<float> l_cgx(10, 5);
    cimg_library::CImg<float> l_cgy(10, 5);
    l_cgx.rand(0.0f, 1.0f);
    l_cgy.rand(0.0f, 1.0f);
    const CImage* const l_gx_p = m_imageFactory_p->createImage(l_cgx);
    const CImage* const l_gy_p = m_imageFactory_p->createImage(l_cgy);
    CImage* l_gx2_p = NULL;
    CImage* l_gxy_p = NULL;
    CImage* l_gy2_p = NULL;
    l_gx_p->triSqr(l_gx_p, l_gy_p, l_gx2_p, l_gxy_p, l_gy2_p);
    compareImages(l_cgx.get_mul(l_cgx), l_gx2_p->asCimg());
    compareImages(l_cgx.get_mul(l_cgy), l_gxy_p->asCimg());
    compareImages(l_cgy.get_mul(l_cgy), l_gy2_p->asCimg());
    delete l_gx_p;
    delete l_gy_p;
    delete l_gx2_p;
    delete l_gxy_p;
    delete l_gy2_p;
}

void CImageTest::lambda2Test()
{
    cimg_library::CImg<float> l_cNxx(10, 5);
    cimg_library::CImg<float> l_cNxy(10, 5);
    cimg_library::CImg<float> l_cNyy(10, 5);
    l_cNxx.rand(0.0f, 1.0f);
    l_cNxy.rand(0.0f, 1.0f);
    l_cNyy.rand(0.0f, 1.0f);
    const CImage* const l_Nxx_p = m_imageFactory_p->createImage(l_cNxx);
    const CImage* const l_Nxy_p = m_imageFactory_p->createImage(l_cNxy);
    const CImage* const l_Nyy_p = m_imageFactory_p->createImage(l_cNyy);
    const float l_M = 1.23f;
    CImage* l_lambda2_p = l_Nxx_p->lambda2(l_M, l_Nxx_p, l_Nxy_p, l_Nyy_p);
    cimg_library::CImg<float> l_cTraceHalf = (l_cNxx + l_cNyy) / 2.0f;
    cimg_library::CImg<float> l_cDet = l_cNxx.get_mul(l_cNyy) - l_cNxy.get_sqr();
    compareImages((l_cTraceHalf - (l_cTraceHalf.get_sqr() - l_cDet).get_sqrt()) * l_M, l_lambda2_p->asCimg());
    delete l_Nxx_p;
    delete l_Nxy_p;
    delete l_Nyy_p;
    delete l_lambda2_p;
}

void CImageTest::triSqrAlphaTest()
{
    cimg_library::CImg<float> l_cgx(10, 5);
    cimg_library::CImg<float> l_cgy(10, 5);
    l_cgx.rand(0.0f, 1.0f);
    l_cgy.rand(0.0f, 1.0f);
    const CImage* const l_gx_p = m_imageFactory_p->createImage(l_cgx);
    const CImage* const l_gy_p = m_imageFactory_p->createImage(l_cgy);
    CImage* l_gx2a_p = NULL;
    CImage* l_2gxya_p = NULL;
    CImage* l_gy2a_p = NULL;
    const float l_alpha = 2.146754979953f;
    l_gx_p->triSqrAlpha(l_alpha, l_gx_p, l_gy_p, l_gx2a_p, l_2gxya_p, l_gy2a_p);
    cimg_library::CImg<float> l_cgxa = l_cgx * std::cos(l_alpha) + l_cgy * std::sin(l_alpha);
    cimg_library::CImg<float> l_cgya = l_cgy * std::cos(l_alpha) - l_cgx * std::sin(l_alpha);
    compareImages(l_cgxa.get_mul(l_cgxa)       , l_gx2a_p->asCimg());
    compareImages(l_cgxa.get_mul(l_cgya) * 2.0f, l_2gxya_p->asCimg());
    compareImages(l_cgya.get_mul(l_cgya)       , l_gy2a_p->asCimg());
    delete l_gx_p;
    delete l_gy_p;
    delete l_gx2a_p;
    delete l_2gxya_p;
    delete l_gy2a_p;
}

void CImageTest::triSumTest()
{
    cimg_library::CImg<float> l_ca(10, 5);
    cimg_library::CImg<float> l_cb(10, 5);
    cimg_library::CImg<float> l_cc(10, 5);
    l_ca.rand(0.0f, 1.0f);
    l_cb.rand(0.0f, 1.0f);
    l_cc.rand(0.0f, 1.0f);
    const CImage* const l_a_p = m_imageFactory_p->createImage(l_ca);
    const CImage* const l_b_p = m_imageFactory_p->createImage(l_cb);
    const CImage* const l_c_p = m_imageFactory_p->createImage(l_cc);
    CImage* l_sum_p = l_a_p->triSum(l_a_p, l_b_p, l_c_p);
    compareImages(l_ca + l_cb + l_cc, l_sum_p->asCimg());
    delete l_a_p;
    delete l_b_p;
    delete l_c_p;
    delete l_sum_p;
}

void CImageTest::precisionTest()
{
    cimg_library::CImg<float> l_cLambda2(10, 5);
    cimg_library::CImg<float> l_cOmega(10, 5);
    l_cLambda2.rand(0.0f, 1.0f);
    l_cOmega.rand(0.1f, 1.0f);
    const CImage* const l_lambda2_p = m_imageFactory_p->createImage(l_cLambda2);
    const CImage* const l_omega_p = m_imageFactory_p->createImage(l_cOmega);
    const float l_factor = 1.23f;
    CImage* l_precision_p = l_lambda2_p->precision(l_factor, l_lambda2_p, l_omega_p);
    compareImages(l_cLambda2.get_div(l_cOmega) * l_factor, l_precision_p->asCimg());
    delete l_lambda2_p;
    delete l_omega_p;
    delete l_precision_p;
}

void CImageTest::bestOmegaTest()
{
    cimg_library::CImg<float> l_cOmega0(10, 5);
    cimg_library::CImg<float> l_cOmega60(10, 5);
    cimg_library::CImg<float> l_cOmega120(10, 5);
    l_cOmega0.rand(0.0f, 1.0f);
    l_cOmega60.rand(0.0f, 1.0f);
    l_cOmega120.rand(0.0f, 1.0f);
    const CImage* const l_omega0_p   = m_imageFactory_p->createImage(l_cOmega0);
    const CImage* const l_omega60_p  = m_imageFactory_p->createImage(l_cOmega60);
    const CImage* const l_omega120_p = m_imageFactory_p->createImage(l_cOmega120);
    CImage* l_omegaMin_p = l_omega0_p->bestOmega(l_omega0_p, l_omega60_p, l_omega120_p);
    compareImages((l_cOmega0 + l_cOmega60 + l_cOmega120) / 3.0f - (2.0f / 3.0f) * sqrt(
                l_cOmega0.get_sqr() + l_cOmega60.get_sqr() + l_cOmega120.get_sqr() -
                l_cOmega0.get_mul(l_cOmega60) - l_cOmega60.get_mul(l_cOmega120) - l_cOmega120.get_mul(l_cOmega0)
                ), l_omegaMin_p->asCimg());
    delete l_omega0_p;
    delete l_omega60_p;
    delete l_omega120_p;
    delete l_omegaMin_p;
}

void CImageTest::downsampleTest()
{
    cimg_library::CImg<float> l_cimg(1024, 1024);
    l_cimg.rand(0.0f, 1.0f);
    const CImage* l_result_p = m_imageFactory_p->createImage(l_cimg)->downsample();
    cimg_library::CImg<float> l_cresult = l_result_p->asCimg();
    cimg_forXY(l_cresult, x, y) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(l_cresult(x, y), l_cimg(x * 2, y * 2), 1e-5);
    }
    delete l_result_p;
}

void CImageTest::cloneTest()
{
    cimg_library::CImg<float> l_cimg(10, 5);
    l_cimg.rand(0.0f, 1.0f);
    const CImage* const l_img_p = m_imageFactory_p->createImage(l_cimg);
    const CImage* const l_clone_p = l_img_p->clone();
    compareImages(l_cimg, l_clone_p->asCimg());
    delete l_clone_p;
    delete l_img_p;
}

void CImageTest::loadTest()
{
    CImage* l_image_p = m_imageFactory_p->createImage();
    l_image_p->load("../examples/check.png");
    cimg_library::CImg<float> l_cimg = l_image_p->asCimg();
    CPPUNIT_ASSERT(l_cimg.min() >= 0.0f);
    CPPUNIT_ASSERT(l_cimg.min() <= 1.0f);
    CPPUNIT_ASSERT(l_cimg.max() >= 0.0f);
    CPPUNIT_ASSERT(l_cimg.max() <= 1.0f);
    CPPUNIT_ASSERT(l_cimg.variance() >= 0.0f);
    delete l_image_p;
}

void CImageTest::findLocalMaxTest()
{
    // generate CImg images
    std::list<CFeature> l_features;
    cimg_library::CImg<float> l_cimgP(     101, 101, 1, 1,  0.0f);
    cimg_library::CImg<float> l_cimgPBelow(101, 101, 1, 1,  0.0f);
    cimg_library::CImg<float> l_cimgPAbove(101, 101, 1, 1,  0.0f);
    cimg_library::CImg<float> l_cimgL2(    101, 101, 1, 1, 42.0f);
    l_cimgP(50, 50) = 1.0f;

    // transfer to CImage images
    const CImage* const l_p_p      = m_imageFactory_p->createImage(l_cimgP);
    const CImage* const l_pBelow_p = m_imageFactory_p->createImage(l_cimgPBelow);
    const CImage* const l_pAbove_p = m_imageFactory_p->createImage(l_cimgPAbove);
    const CImage* const l_l2_p     = m_imageFactory_p->createImage(l_cimgL2);

    // test finding local maxima
    l_p_p->findLocalMax(&l_features, l_pBelow_p, l_pAbove_p, l_l2_p, 30.0f, 4, 2);
    CPPUNIT_ASSERT_EQUAL(1, (int) l_features.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0f * 2.0f, l_features.begin()->x,         1e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0f * 2.0f, l_features.begin()->y,         1e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(30.0f * 2.0f, l_features.begin()->sigma,     1e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0f * 4.0f, l_features.begin()->precision, 1e-5);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0f / 4.0f, l_features.begin()->lambda2,   1e-5);

    // clean up
    delete l_p_p;
    delete l_pBelow_p;
    delete l_pAbove_p;
    delete l_l2_p;
}

void CImageTest::tearDown()
{
    delete m_imageFactory_p;
    m_imageFactory_p = NULL;
}

}

