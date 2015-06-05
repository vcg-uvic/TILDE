#include "CImageCpu.h"

namespace SFOP {

CImageCpu::CImageCpu(
        const EFilterNames f_filterName,
        const float f_sigma)
{
    unsigned short int l_kSize = 4.0f * f_sigma;
    this->assign(2 * l_kSize + 1);
    const float f = 0.398942280401433 / f_sigma;
    switch (f_filterName) {
        case G:
            cimg_forX(*this, x) {
                const float dX = x - l_kSize;
                (*this)[x] = f * std::exp(-0.5f * dX * dX / f_sigma / f_sigma);
            }
            break;
        case Gx:
            cimg_forX(*this, x) {
                const float dX = x - l_kSize;
                (*this)[x] = f * 0.5f * (
                    std::exp(-0.5f * (dX + 1) * (dX + 1) / f_sigma / f_sigma) -
                    std::exp(-0.5f * (dX - 1) * (dX - 1) / f_sigma / f_sigma));
            }
            break;
        case xG:
            cimg_forX(*this, x) {
                const float dX = x - l_kSize;
                (*this)[x] = f * std::exp(-0.5f * dX * dX / f_sigma / f_sigma) * dX;
            }
            break;
        case x2G:
            cimg_forX(*this, x) {
                const float dX = x - l_kSize;
                (*this)[x] = f * std::exp(-0.5f * dX * dX / f_sigma / f_sigma) * dX * dX;
            }
            break;
    }
}

void CImageCpu::load(const char f_filename[])
{
    this->cimg_library::CImg<float>::load(f_filename);
    if (this->spectrum() > 1) {
        this->RGBtoHSV();
        this->channel(2);
    }
    while (this->max() > 1) this->operator/=(255.0f);
}

CImage* CImageCpu::conv(
        const CImage *f_rowFilter_p,
        const CImage *f_colFilter_p)
{
    cimg_library::CImg<float> l_rowFilter = f_rowFilter_p->asCimg();
    cimg_library::CImg<float> l_colFilter = f_colFilter_p->asCimg();
    if (l_rowFilter.height() > 1) l_rowFilter.transpose();
    if (l_colFilter.width()  > 1) l_colFilter.transpose();
    this->cimg_library::CImg<float>::convolve(l_rowFilter);
    this->cimg_library::CImg<float>::convolve(l_colFilter);
    return this;
}

void CImageCpu::triSqr(
        const CImage* f_gx_p,
        const CImage* f_gy_p,
        CImage* &f_gx2_p,
        CImage* &f_gxy_p,
        CImage* &f_gy2_p) const
{
    f_gx2_p = f_gx_p->clone();
    f_gxy_p = f_gx_p->clone();
    f_gy2_p = f_gy_p->clone();
    ((CImageCpu*)f_gx2_p)->cimg_library::CImg<float>::sqr();
    ((CImageCpu*)f_gxy_p)->cimg_library::CImg<float>::mul(f_gy_p->asCimg());
    ((CImageCpu*)f_gy2_p)->cimg_library::CImg<float>::sqr();
}

CImage* CImageCpu::lambda2(
        const float f_M,
        const CImage* f_Nxx_p,
        const CImage* f_Nxy_p,
        const CImage* f_Nyy_p) const
{
    CImageCpu* l_traceHalf_p = dynamic_cast<CImageCpu*>(f_Nxx_p->clone());
    l_traceHalf_p->cimg_library::CImg<float>::operator+=(f_Nyy_p->asCimg());
    l_traceHalf_p->cimg_library::CImg<float>::operator/=(2.0f);

    CImageCpu* l_det_p = dynamic_cast<CImageCpu*>(f_Nxx_p->clone());
    l_det_p->cimg_library::CImg<float>::mul(f_Nyy_p->asCimg());
    l_det_p->cimg_library::CImg<float>::operator-=(f_Nxy_p->asCimg().sqr());

    CImageCpu* l_lambda2_p = dynamic_cast<CImageCpu*>(l_traceHalf_p->clone());
    l_traceHalf_p->cimg_library::CImg<float>::sqr();
    l_traceHalf_p->cimg_library::CImg<float>::operator-=(
        *(const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(l_det_p)));
    l_traceHalf_p->cimg_library::CImg<float>::sqrt();
    l_lambda2_p->cimg_library::CImg<float>::operator-=(
        *(const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(l_traceHalf_p)));
    l_lambda2_p->cimg_library::CImg<float>::operator*=(f_M);

    delete l_traceHalf_p;
    delete l_det_p;

    return l_lambda2_p;
}

void CImageCpu::triSqrAlpha(
        const float f_alpha,
        const CImage* f_gx_p,
        const CImage* f_gy_p,
        CImage* &f_gx2a_p,
        CImage* &f_2gxya_p,
        CImage* &f_gy2a_p) const
{
    CImageCpu* l_gxa_p = dynamic_cast<CImageCpu*>(f_gx_p->clone());
    l_gxa_p->cimg_library::CImg<float>::operator*=(std::cos(f_alpha));
    l_gxa_p->cimg_library::CImg<float>::operator+=(f_gy_p->asCimg().operator*=(std::sin(f_alpha)));

    CImageCpu* l_gya_p = dynamic_cast<CImageCpu*>(f_gy_p->clone());
    l_gya_p->cimg_library::CImg<float>::operator*=(std::cos(f_alpha));
    l_gya_p->cimg_library::CImg<float>::operator-=(f_gx_p->asCimg().operator*=(std::sin(f_alpha)));

    f_gx2a_p  = l_gxa_p->clone();
    f_2gxya_p = l_gxa_p->clone();
    f_gy2a_p  = l_gya_p->clone();
    ((CImageCpu*)f_gx2a_p )->cimg_library::CImg<float>::sqr();
    ((CImageCpu*)f_2gxya_p)->cimg_library::CImg<float>::mul(l_gya_p->asCimg());
    ((CImageCpu*)f_2gxya_p)->cimg_library::CImg<float>::operator*=(2.0f);
    ((CImageCpu*)f_gy2a_p )->cimg_library::CImg<float>::sqr();

    delete l_gxa_p;
    delete l_gya_p;
}

CImage* CImageCpu::triSum(
        const CImage* f_a_p,
        const CImage* f_b_p,
        const CImage* f_c_p) const
{
    CImageCpu* l_sum_p = dynamic_cast<CImageCpu*>(f_a_p->clone());
    l_sum_p->cimg_library::CImg<float>::operator+=(f_b_p->asCimg());
    l_sum_p->cimg_library::CImg<float>::operator+=(f_c_p->asCimg());
    return l_sum_p;
}

CImage* CImageCpu::precision(
        const float f_factor,
        const CImage* f_lambda2_p,
        const CImage* f_omega_p) const
{
    CImageCpu* l_precision_p = dynamic_cast<CImageCpu*>(f_lambda2_p->clone());
    l_precision_p->cimg_library::CImg<float>::div(f_omega_p->asCimg());
    l_precision_p->cimg_library::CImg<float>::operator*=(f_factor);
    return l_precision_p;
}

CImage* CImageCpu::bestOmega(
        const CImage* f_omega0_p,
        const CImage* f_omega60_p,
        const CImage* f_omega120_p) const
{
    CImageCpu* l_omegaBest_p = dynamic_cast<CImageCpu*>(f_omega0_p->clone());
    l_omegaBest_p->cimg_library::CImg<float>::operator+=(f_omega60_p->asCimg());
    l_omegaBest_p->cimg_library::CImg<float>::operator+=(f_omega120_p->asCimg());
    CImageCpu* l_sqrt_p  = dynamic_cast<CImageCpu*>(f_omega0_p->clone());
    CImageCpu* l_sqrt2_p = dynamic_cast<CImageCpu*>(f_omega60_p->clone());
    CImageCpu* l_sqrt3_p = dynamic_cast<CImageCpu*>(f_omega120_p->clone());
    CImageCpu* l_sqrt4_p = dynamic_cast<CImageCpu*>(f_omega0_p->clone());
    CImageCpu* l_sqrt5_p = dynamic_cast<CImageCpu*>(f_omega60_p->clone());
    CImageCpu* l_sqrt6_p = dynamic_cast<CImageCpu*>(f_omega120_p->clone());
    l_sqrt_p->cimg_library::CImg<float>::sqr();
    l_sqrt2_p->cimg_library::CImg<float>::sqr();
    l_sqrt3_p->cimg_library::CImg<float>::sqr();
    l_sqrt4_p->cimg_library::CImg<float>::mul(f_omega60_p->asCimg());
    l_sqrt5_p->cimg_library::CImg<float>::mul(f_omega120_p->asCimg());
    l_sqrt6_p->cimg_library::CImg<float>::mul(f_omega0_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::operator+=(l_sqrt2_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::operator+=(l_sqrt3_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::operator-=(l_sqrt4_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::operator-=(l_sqrt5_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::operator-=(l_sqrt6_p->asCimg());
    l_sqrt_p->cimg_library::CImg<float>::sqrt();
    l_sqrt_p->cimg_library::CImg<float>::operator*=(2.0f);
    l_omegaBest_p->cimg_library::CImg<float>::operator-=(l_sqrt_p->asCimg());
    l_omegaBest_p->cimg_library::CImg<float>::operator/=(3.0f);

    delete l_sqrt_p;
    delete l_sqrt2_p;
    delete l_sqrt3_p;
    delete l_sqrt4_p;
    delete l_sqrt5_p;
    delete l_sqrt6_p;

    return l_omegaBest_p;
}

CImage* CImageCpu::downsample()
{
    cimg_library::CImg<float>::resize(-50, -50);
    return this;
}

CImage* CImageCpu::clone() const
{
    return new CImageCpu(cimg_library::CImg<float>(*this));
}

cimg_library::CImg<float> CImageCpu::asCimg() const
{
    return cimg_library::CImg<float>(*this);
}

cimg_library::CImg<float> CImageCpu::computeGradient(
        const cimg_library::CImg<float> &f_cube)
{
    const float l_g[3] = {
        - f_cube[ 0] / 32.0f + f_cube[ 2] / 32.0f - f_cube[ 3] / 16.0f
        + f_cube[ 5] / 16.0f - f_cube[ 6] / 32.0f + f_cube[ 8] / 32.0f
        - f_cube[ 9] / 16.0f + f_cube[11] / 16.0f - f_cube[12] /  8.0f
        + f_cube[14] /  8.0f - f_cube[15] / 16.0f + f_cube[17] / 16.0f
        - f_cube[18] / 32.0f + f_cube[20] / 32.0f - f_cube[21] / 16.0f
        + f_cube[23] / 16.0f - f_cube[24] / 32.0f + f_cube[26] / 32.0f,
        - f_cube[ 0] / 32.0f - f_cube[ 1] / 16.0f - f_cube[ 2] / 32.0f
        + f_cube[ 6] / 32.0f + f_cube[ 7] / 16.0f + f_cube[ 8] / 32.0f
        - f_cube[ 9] / 16.0f - f_cube[10] /  8.0f - f_cube[11] / 16.0f
        + f_cube[15] / 16.0f + f_cube[16] /  8.0f + f_cube[17] / 16.0f
        - f_cube[18] / 32.0f - f_cube[19] / 16.0f - f_cube[20] / 32.0f
        + f_cube[24] / 32.0f + f_cube[25] / 16.0f + f_cube[26] / 32.0f,
        - f_cube[ 0] / 32.0f - f_cube[ 1] / 16.0f - f_cube[ 2] / 32.0f
        - f_cube[ 3] / 16.0f - f_cube[ 4] /  8.0f - f_cube[ 5] / 16.0f
        - f_cube[ 6] / 32.0f - f_cube[ 7] / 16.0f - f_cube[ 8] / 32.0f
        + f_cube[18] / 32.0f + f_cube[19] / 16.0f + f_cube[20] / 32.0f
        + f_cube[21] / 16.0f + f_cube[22] /  8.0f + f_cube[23] / 16.0f
        + f_cube[24] / 32.0f + f_cube[25] / 16.0f + f_cube[26] / 32.0f};
    return cimg_library::CImg<float>(l_g, 1, 3);
}

cimg_library::CImg<float> CImageCpu::computeHessian(
        const cimg_library::CImg<float> &f_cube)
{
    const float l_HM[9] = {
        + f_cube[ 0] / 16.0f - f_cube[ 1] /  8.0f + f_cube[ 2] / 16.0f
        + f_cube[ 3] /  8.0f - f_cube[ 4] /  4.0f + f_cube[ 5] /  8.0f
        + f_cube[ 6] / 16.0f - f_cube[ 7] /  8.0f + f_cube[ 8] / 16.0f
        + f_cube[ 9] /  8.0f - f_cube[10] /  4.0f + f_cube[11] /  8.0f
        + f_cube[12] /  4.0f - f_cube[13] /  2.0f + f_cube[14] /  4.0f
        + f_cube[15] /  8.0f - f_cube[16] /  4.0f + f_cube[17] /  8.0f
        + f_cube[18] / 16.0f - f_cube[19] /  8.0f + f_cube[20] / 16.0f
        + f_cube[21] /  8.0f - f_cube[22] /  4.0f + f_cube[23] /  8.0f
        + f_cube[24] / 16.0f - f_cube[25] /  8.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f - f_cube[ 2] / 16.0f - f_cube[ 6] / 16.0f
        + f_cube[ 8] / 16.0f + f_cube[ 9] /  8.0f - f_cube[11] /  8.0f
        - f_cube[15] /  8.0f + f_cube[17] /  8.0f + f_cube[18] / 16.0f
        - f_cube[20] / 16.0f - f_cube[24] / 16.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f - f_cube[ 2] / 16.0f + f_cube[ 3] /  8.0f
        - f_cube[ 5] /  8.0f + f_cube[ 6] / 16.0f - f_cube[ 8] / 16.0f
        - f_cube[18] / 16.0f + f_cube[20] / 16.0f - f_cube[21] /  8.0f
        + f_cube[23] /  8.0f - f_cube[24] / 16.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f - f_cube[ 2] / 16.0f - f_cube[ 6] / 16.0f
        + f_cube[ 8] / 16.0f + f_cube[ 9] /  8.0f - f_cube[11] /  8.0f
        - f_cube[15] /  8.0f + f_cube[17] /  8.0f + f_cube[18] / 16.0f
        - f_cube[20] / 16.0f - f_cube[24] / 16.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f + f_cube[ 1] /  8.0f + f_cube[ 2] / 16.0f
        - f_cube[ 3] /  8.0f - f_cube[ 4] /  4.0f - f_cube[ 5] /  8.0f
        + f_cube[ 6] / 16.0f + f_cube[ 7] /  8.0f + f_cube[ 8] / 16.0f
        + f_cube[ 9] /  8.0f + f_cube[10] /  4.0f + f_cube[11] /  8.0f
        - f_cube[12] /  4.0f - f_cube[13] /  2.0f - f_cube[14] /  4.0f
        + f_cube[15] /  8.0f + f_cube[16] /  4.0f + f_cube[17] /  8.0f
        + f_cube[18] / 16.0f + f_cube[19] /  8.0f + f_cube[20] / 16.0f
        - f_cube[21] /  8.0f - f_cube[22] /  4.0f - f_cube[23] /  8.0f
        + f_cube[24] / 16.0f + f_cube[25] /  8.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f + f_cube[ 1] /  8.0f + f_cube[ 2] / 16.0f
        - f_cube[ 6] / 16.0f - f_cube[ 7] /  8.0f - f_cube[ 8] / 16.0f
        - f_cube[18] / 16.0f - f_cube[19] /  8.0f - f_cube[20] / 16.0f
        + f_cube[24] / 16.0f + f_cube[25] /  8.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f - f_cube[ 2] / 16.0f + f_cube[ 3] /  8.0f
        - f_cube[ 5] /  8.0f + f_cube[ 6] / 16.0f - f_cube[ 8] / 16.0f
        - f_cube[18] / 16.0f + f_cube[20] / 16.0f - f_cube[21] /  8.0f
        + f_cube[23] /  8.0f - f_cube[24] / 16.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f + f_cube[ 1] /  8.0f + f_cube[ 2] / 16.0f
        - f_cube[ 6] / 16.0f - f_cube[ 7] /  8.0f - f_cube[ 8] / 16.0f
        - f_cube[18] / 16.0f - f_cube[19] /  8.0f - f_cube[20] / 16.0f
        + f_cube[24] / 16.0f + f_cube[25] /  8.0f + f_cube[26] / 16.0f,
        + f_cube[ 0] / 16.0f + f_cube[ 1] /  8.0f + f_cube[ 2] / 16.0f
        + f_cube[ 3] /  8.0f + f_cube[ 4] /  4.0f + f_cube[ 5] /  8.0f
        + f_cube[ 6] / 16.0f + f_cube[ 7] /  8.0f + f_cube[ 8] / 16.0f
        - f_cube[ 9] /  8.0f - f_cube[10] /  4.0f - f_cube[11] /  8.0f
        - f_cube[12] /  4.0f - f_cube[13] /  2.0f - f_cube[14] /  4.0f
        - f_cube[15] /  8.0f - f_cube[16] /  4.0f - f_cube[17] /  8.0f
        + f_cube[18] / 16.0f + f_cube[19] /  8.0f + f_cube[20] / 16.0f
        + f_cube[21] /  8.0f + f_cube[22] /  4.0f + f_cube[23] /  8.0f
        + f_cube[24] / 16.0f + f_cube[25] /  8.0f + f_cube[26] / 16.0f};
    return cimg_library::CImg<float>(l_HM, 3, 3);
}

bool CImageCpu::checkNegativeDefinite(const cimg_library::CImg<float> &f_m)
{
    return
        (f_m(0, 0) < 0.0f) &&
        (f_m(0, 0) * f_m(1, 1) - f_m(1, 0) * f_m(0, 1) > 0.0f) &&
        (f_m(0, 0) * (f_m(1, 1) * f_m(2, 2) - f_m(2, 1) * f_m(1, 2)) -
         f_m(0, 1) * (f_m(1, 0) * f_m(2, 2) - f_m(2, 0) * f_m(1, 2)) +
         f_m(0, 2) * (f_m(1, 0) * f_m(2, 1) - f_m(2, 0) * f_m(1, 1)) < 0.0f);
}

void CImageCpu::findLocalMax(
        std::list<CFeature>* f_features_p,
        const CImage* f_below_p,
        const CImage* f_above_p,
        const CImage* f_lambda2_p,
        const float f_sigma,
        const unsigned int f_numLayers,
        const unsigned int f_factor) const
{
    // get images as CImg
    const cimg_library::CImg<float>* l_above_p   = (const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(f_above_p));
    const cimg_library::CImg<float>* l_this_p    = (const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(this));
    const cimg_library::CImg<float>* l_below_p   = (const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(f_below_p));
    const cimg_library::CImg<float>* l_lambda2_p = (const cimg_library::CImg<float>*)(dynamic_cast<const CImageCpu*>(f_lambda2_p));

    // collect features as local maxima from the whole image
    cimg_for_insideXY(*l_this_p, x, y, 1) {

        // check for local maximum
        float p = (*l_this_p)(x, y);
        bool fail = false;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (p < (*l_below_p)(x + dx, y + dy) ||
                    p < (*l_above_p)(x + dx, y + dy) ||
                    p < (*l_this_p)( x + dx, y + dy)) {
                    fail = true;
                    break;
                }
            }
            if (fail) break;
        }
        if (fail) continue;

        // copy 3x3x3 neighborhood of pixel
        cimg_library::CImg<float> P27(3, 3, 3);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                P27(dx + 1, dy + 1, 0) = (*l_below_p)(x + dx, y + dy);
                P27(dx + 1, dy + 1, 1) = (*l_this_p)( x + dx, y + dy);
                P27(dx + 1, dy + 1, 2) = (*l_above_p)(x + dx, y + dy);
            }
        }

        // compute the negative definite Hessian
        cimg_library::CImg<float> H = computeHessian(P27);
        if (!checkNegativeDefinite(H)) continue;

        // compute and check update
        cimg_library::CImg<float> g = computeGradient(P27);
        cimg_library::CImg<float> update = -g.solve(H);
        if ((update(0, 0) != update(0, 0)) ||
            (update(0, 1) != update(0, 1)) ||
            (update(0, 2) != update(0, 2)) ||
            (update(0, 0) * update(0, 0) +
             update(0, 1) * update(0, 1) +
             update(0, 2) * update(0, 2) > 1.0f)) continue;

        // add update
        const float l_x = x + update(0, 0);
        const float l_y = y + update(0, 1);
        const float l_s =     update(0, 2);
        const float l_sigma = f_sigma * std::pow(2.0f, l_s / f_numLayers);
        const float l_p = (*l_this_p)(x, y) + 0.5f * g.dot(update);
        const float l_l2 = (*l_lambda2_p)(floor(l_x + 0.5f), floor(l_y + 0.5f));
        f_features_p->push_back(CFeature(
                    l_x * (float) f_factor,
                    l_y * (float) f_factor,
                    l_sigma * (float) f_factor,
                    l_l2 / (float) f_factor / (float) f_factor,
                    l_p * (float) f_factor * (float) f_factor));

    }
}

}

