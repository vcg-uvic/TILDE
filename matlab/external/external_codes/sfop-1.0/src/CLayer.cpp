#include "CLayer.h"

namespace SFOP {

CLayer::CLayer(
        const CImageFactoryAbstract* f_factory_p,
        const CImage* f_img_p,
        const int f_layer,
        const unsigned int f_octave,
        const unsigned int f_numLayers,
        const float f_type) : m_factory_p(f_factory_p),
    m_layer(f_layer), m_octave(f_octave), m_numLayers(f_numLayers)
{
    // scales w.r.t. image at current octave: layer index involved, but not octave
    const float l_sigma = pow(2.0f, 1.0f + (1.0f + m_layer) / m_numLayers);
    const float l_tau = l_sigma / 3.0f;
    const float l_M = 12.0f * l_sigma * l_sigma + 1.0f;
    const float l_R = l_M - 3.0f;

    // differentiation
    CImage* l_Gtau_p = m_factory_p->createImage(CImage::G, l_tau);
    CImage* l_Gxtau_p = m_factory_p->createImage(CImage::Gx, l_tau);
    CImage* l_gx_p = f_img_p->clone()->conv(l_Gxtau_p, l_Gtau_p); // in-place
    CImage* l_gy_p = f_img_p->clone()->conv(l_Gtau_p, l_Gxtau_p); // in-place
    delete l_Gtau_p;
    delete l_Gxtau_p;
    l_Gtau_p = NULL;
    l_Gxtau_p = NULL;

    // structure tensor (not yet scaled by M!)
    CImage* l_gxgx_p = NULL;
    CImage* l_gxgy_p = NULL;
    CImage* l_gygy_p = NULL;
    f_img_p->triSqr(l_gx_p, l_gy_p, l_gxgx_p, l_gxgy_p, l_gygy_p);

    CImage* l_Nxx_p = l_gxgx_p;
    l_gxgx_p = NULL;
    CImage* l_Gsigma_p = m_factory_p->createImage(CImage::G, l_sigma);
    l_Nxx_p->conv(l_Gsigma_p, l_Gsigma_p);
    CImage* l_Nxy_p = l_gxgy_p;
    l_gxgy_p = NULL;
    l_Nxy_p->conv(l_Gsigma_p, l_Gsigma_p);
    CImage* l_Nyy_p = l_gygy_p;
    l_gygy_p = NULL;
    l_Nyy_p->conv(l_Gsigma_p, l_Gsigma_p);

    // smaller eigenvalue
    m_lambda2_p = f_img_p->lambda2(l_M, l_Nxx_p, l_Nxy_p, l_Nyy_p);
    delete l_Nxx_p;
    delete l_Nxy_p;
    delete l_Nyy_p;
    l_Nxx_p = NULL;
    l_Nxy_p = NULL;
    l_Nyy_p = NULL;

    // model error
    const unsigned int l_numAngles = f_type < 0.0f ? 3 : 1;
    CImage* l_omegas_p[l_numAngles];
    CImage* l_gxx0_p  = NULL;
    CImage* l_2gxy0_p = NULL;
    CImage* l_gyy0_p  = NULL;
    CImage* l_xGsigma_p = m_factory_p->createImage(CImage::xG, l_sigma);
    CImage* l_x2Gsigma_p = m_factory_p->createImage(CImage::x2G, l_sigma);
    for (unsigned int a = 0; a < l_numAngles; ++a) {
        const float l_angle = f_type < 0.0f ? (float) a * M_PI / 3.0f : f_type / 180.0f * M_PI;
        f_img_p->triSqrAlpha(l_angle, l_gx_p, l_gy_p, l_gxx0_p, l_2gxy0_p, l_gyy0_p);
        l_gxx0_p->conv(l_x2Gsigma_p, l_Gsigma_p);
        l_2gxy0_p->conv(l_xGsigma_p, l_xGsigma_p);
        l_gyy0_p->conv(l_Gsigma_p, l_x2Gsigma_p);
        l_omegas_p[a] = f_img_p->triSum(l_gxx0_p, l_2gxy0_p, l_gyy0_p);
        delete l_gxx0_p;
        delete l_2gxy0_p;
        delete l_gyy0_p;
    }
    l_gxx0_p  = NULL;
    l_2gxy0_p = NULL;
    l_gyy0_p  = NULL;
    delete l_gx_p;
    delete l_gy_p;
    l_gx_p = NULL;
    l_gy_p = NULL;
    delete l_Gsigma_p;
    delete l_xGsigma_p;
    delete l_x2Gsigma_p;
    l_Gsigma_p   = NULL;
    l_xGsigma_p  = NULL;
    l_x2Gsigma_p = NULL;
    CImage* l_omega_p;
    if (f_type < 0.0f) {
        l_omega_p = f_img_p->bestOmega(l_omegas_p[0], l_omegas_p[1], l_omegas_p[2]);
        for (unsigned int a = 0; a < l_numAngles; ++a) {
            delete l_omegas_p[a];
            l_omegas_p[a] = NULL;
        }
    }
    else {
        l_omega_p = l_omegas_p[0];
        l_omegas_p[0] = NULL;
    }

    // precision
    m_precision_p = f_img_p->precision(l_R / l_M, m_lambda2_p, l_omega_p);
    delete l_omega_p;
    l_omega_p = NULL;
}

CLayer::~CLayer()
{
    if (m_lambda2_p == NULL) {
        std::cerr << "Error in CLayer::~CLayer(): m_lambda2_p is NULL." << std::endl;
        exit(1);
    }
    delete m_lambda2_p;
    m_lambda2_p = NULL;
    if (m_precision_p == NULL) {
        std::cerr << "Error in CLayer::~CLayer(): m_precision_p is NULL." << std::endl;
        exit(1);
    }
    delete m_precision_p;
    m_precision_p = NULL;
}

void CLayer::detect(
        std::list<CFeature>* f_features_p,
        const CLayer* f_below_p,
        const CLayer* f_above_p) const
{
    m_precision_p->findLocalMax(
            f_features_p,
            f_below_p->precision_p(),
            f_above_p->precision_p(),
            m_lambda2_p,
            pow(2.0f, 1.0f + (1.0f + m_layer) / m_numLayers),
            m_numLayers,
            pow(2.0f, (float) m_octave));
}

}

