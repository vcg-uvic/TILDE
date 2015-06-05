#include "CSfop.h"

namespace SFOP {

CSfop::CSfop(
        const CImageFactoryAbstract* f_factory_p,
        const char f_filename[]) : m_factory_p(f_factory_p)
{
    m_image_p = m_factory_p->createImage();
    m_image_p->load(f_filename);
    m_features_p = new std::list<CFeature>();
}

CSfop::~CSfop()
{
    delete m_image_p;
}

bool sortByPrecision(
        const CFeature &left,
        const CFeature &right)
{
    return left.precision > right.precision;
}

double CSfop::chi2inv(
        const double f_p,
        const double f_v)
{
    const double l_a = f_v / 2.0;
    const double l_sigsq = log(1 + l_a) - log(l_a);
    double l_q = exp(log(l_a) - l_sigsq / 2.0 - sqrt(2.0 * l_sigsq) * erfcore(2.0 * f_p));
    double l_h = DBL_MAX;
    while (std::abs(l_h) > 1e-12 * l_q) {
        l_h = (gammainc(l_q, l_a) - f_p) / exp((l_a - 1.0) * log(l_q) - l_q - gammaln(l_a));
        l_q = std::max(l_q / 10.0, std::min(10.0 * l_q, l_q - l_h));
    }
    return 2.0 * l_q;
}

double CSfop::gammaln(
        const double f_x)
{
    const double l_d1 = -5.772156649015328605195174e-1;
    const double l_p1[8] = {
        4.945235359296727046734888e00, 2.018112620856775083915565e02, 
        2.290838373831346393026739e03, 1.131967205903380828685045e04, 
        2.855724635671635335736389e04, 3.848496228443793359990269e04, 
        2.637748787624195437963534e04, 7.225813979700288197698961e03};
    const double l_q1[8] = {
        6.748212550303777196073036e01, 1.113332393857199323513008e03, 
        7.738757056935398733233834e03, 2.763987074403340708898585e04, 
        5.499310206226157329794414e04, 6.161122180066002127833352e04, 
        3.635127591501940507276287e04, 8.785536302431013170870835e03};
    const double l_d2 = 4.227843350984671393993777e-1;
    const double l_p2[8] = {
        4.974607845568932035012064e00, 5.424138599891070494101986e02, 
        1.550693864978364947665077e04, 1.847932904445632425417223e05, 
        1.088204769468828767498470e06, 3.338152967987029735917223e06, 
        5.106661678927352456275255e06, 3.074109054850539556250927e06};
    const double l_q2[8] = {
        1.830328399370592604055942e02, 7.765049321445005871323047e03, 
        1.331903827966074194402448e05, 1.136705821321969608938755e06, 
        5.267964117437946917577538e06, 1.346701454311101692290052e07, 
        1.782736530353274213975932e07, 9.533095591844353613395747e06};
    const double l_p4[8] = {
        1.474502166059939948905062e04, 2.426813369486704502836312e06, 
        1.214755574045093227939592e08, 2.663432449630976949898078e09, 
        2.940378956634553899906876e10, 1.702665737765398868392998e11, 
        4.926125793377430887588120e11, 5.606251856223951465078242e11};
    const double l_q4[8] = {
        2.690530175870899333379843e03, 6.393885654300092398984238e05, 
        4.135599930241388052042842e07, 1.120872109616147941376570e09, 
        1.488613728678813811542398e10, 1.016803586272438228077304e11, 
        3.417476345507377132798597e11, 4.463158187419713286462081e11};
    const double l_c[6] = {
        -1.910444077728000000000000e-3, 8.417138778129500000000000e-4,
        -5.952379913043012000000000e-4, 7.936507935003502480000000e-4,
        -2.777777777777681622553000e-3, 8.333333333333333331554247e-2};
    double l_xnum = 0.0;
    double l_xden = 1.0;
    if (f_x <= 0.5) {
        for (unsigned int i = 0; i < 8; ++i) {
            l_xnum = l_xnum * f_x + l_p1[i];
            l_xden = l_xden * f_x + l_q1[i];
        }
        return -log(f_x) + f_x * (l_d1 + f_x * l_xnum / l_xden);
    }
    else if (f_x <= 0.6796875) {
        for (unsigned int i = 0; i < 8; ++i) {
            l_xnum = l_xnum * (f_x - 1.0) + l_p2[i];
            l_xden = l_xden * (f_x - 1.0) + l_q2[i];
        }
        return -log(f_x) + (f_x - 1.0) * (l_d2 + (f_x - 1.0) * l_xnum / l_xden);
    }
    else if (f_x <= 1.5) {
        for (unsigned int i = 0; i < 8; ++i) {
            l_xnum = l_xnum * (f_x - 1.0) + l_p1[i];
            l_xden = l_xden * (f_x - 1.0) + l_q1[i];
        }
        return (f_x - 1.0) * (l_d1 + (f_x - 1.0) * l_xnum / l_xden);
    }
    else if (f_x <= 4.0) {
        for (unsigned int i = 0; i < 8; ++i) {
            l_xnum = l_xnum * (f_x - 2.0) + l_p2[i];
            l_xden = l_xden * (f_x - 2.0) + l_q2[i];
        }
        return (f_x - 2.0) * (l_d2 + (f_x - 2.0) * l_xnum / l_xden);
    }
    else if (f_x <= 12.0) {
        for (unsigned int i = 0; i < 8; ++i) {
            l_xnum = l_xnum * (f_x - 4.0) + l_p4[i];
            l_xden = l_xden * (f_x - 4.0) - l_q4[i];
        }
        return 1.791759469228055000094023 - (f_x - 4.0) * l_xnum / l_xden;
    }
    else {
        double l_r = 5.7083835261e-3;
        for (unsigned int i = 0; i < 6; ++i) {
            l_r = l_r / f_x / f_x + l_c[i];
        }
        return l_r / f_x + 0.91893853320467274178033 - log(f_x) / 2.0 + f_x * log(f_x) - f_x;
    }
}

double CSfop::gammainc(
        const double f_x,
        const double f_a)
{
    if (f_x < f_a + 1.0) {
        double l_ap = f_a;
        double l_del = 1.0;
        double l_sum = 1.0;
        while (std::abs(l_del) >= 1e-14) {
            l_ap = l_ap + 1.0;
            l_del = f_x * l_del / l_ap;
            l_sum = l_sum + l_del;
        }
        return l_sum * exp(f_a * log(f_x) - f_x - gammaln(f_a + 1.0));
    }
    else {
        double l_a0 = 1.0;
        double l_a1 = f_x;
        double l_b0 = 0.0;
        double l_b1 = 1.0;
        double l_n = 1.0;
        double l_g = l_b1 / l_a1;
        double l_gold = 0.0;
        while (std::abs(l_g - l_gold) >= 1e-14 * l_g) {
            l_gold = l_g;
            l_b0 = (l_b1 + l_b0 * (l_n - f_a)) / l_a1;
            l_a0 = (l_a1 + l_a0 * (l_n - f_a)) / l_a1;
            l_b1 = f_x * l_b0 + l_n / l_a1 * l_b1;
            l_a1 = f_x * l_a0 + l_n;
            l_g = l_b1 / l_a1;
            l_n = l_n + 1.0;
        }
        return 1.0 - exp(f_a * log(f_x) - f_x - gammaln(f_a)) * l_g;
    }
}

double CSfop::erfcore(
        const double f_x)
{
    const double l_c[8] = {
        5.64188496988670089e-1, 8.88314979438837594e00,
        6.61191906371416295e01, 2.98635138197400131e02,
        8.81952221241769090e02, 1.71204761263407058e03,
        2.05107837782607147e03, 1.23033935479799725e03};
    const double l_d[8] = {
        1.57449261107098347e01, 1.17693950891312499e02,
        5.37181101862009858e02, 1.62138957456669019e03,
        3.29079923573345963e03, 4.36261909014324716e03,
        3.43936767414372164e03, 1.23033935480374942e03};
    double l_xnum = f_x * 2.15311535474403846e-8;
    double l_xden = f_x;
    for (unsigned int i = 0; i < 8; ++i) {
        l_xnum = (l_xnum + l_c[i]) * f_x;
        l_xden = (l_xden + l_d[i]) * f_x;
    }
    return l_xnum / l_xden / exp(f_x) / 2.0;
}

void CSfop::detect(
        const unsigned int f_numOctaves,
        const unsigned int f_numLayers,
        const float f_imageNoise,
        const float f_lambdaWeight,
        const float f_precisionThreshold,
        const float f_type)
{
    // clear features
    m_features_p->clear();

    // save original image size
    const unsigned int l_width = m_image_p->width();
    const unsigned int l_height = m_image_p->height();

    // process octaves
    CImage* l_gaussian_p = m_factory_p->createImage(CImage::G, (float) sqrt(0.5f));
    for (unsigned int o = 0; o < f_numOctaves; ++o) {

        // downsample image
        if (o > 0) m_image_p->conv(l_gaussian_p, l_gaussian_p)->downsample();

        // build scale space
        COctave l_octave(m_factory_p, m_image_p, o, f_numLayers, f_type);

        // detect new features
        l_octave.detect(m_features_p);
    }
    delete l_gaussian_p;
    l_gaussian_p = NULL;
    std::cout << m_features_p->size() << " features detected." << std::endl;

    // filter features
    filterFeatures(f_imageNoise, f_lambdaWeight, f_precisionThreshold, l_width, l_height);
    std::cout << m_features_p->size() << " features passed thresholds." << std::endl;

    // suppress non-maxima
    nonMaximaSuppression(f_numLayers);
    std::cout << m_features_p->size() << " features after non-maxima suppression." << std::endl;
}

void CSfop::filterFeatures(
        const float f_imageNoise,
        const float f_lambdaWeight,
        const float f_precisionThreshold,
        const unsigned int f_width,
        const unsigned int f_height)
{
    std::list<CFeature>::iterator l_iter;
    l_iter = m_features_p->begin();
    while (l_iter != m_features_p->end()) {
        const float l_sigma = l_iter->sigma;
        const float l_tau = l_sigma / 3.0f;
        const float l_h = f_imageNoise * f_imageNoise / 16.0f / M_PI / l_tau / l_tau / l_tau / l_tau;
        const float l_Tl2 = l_h * f_lambdaWeight * chi2inv(0.999, 24.0f * l_sigma * l_sigma + 2.0f);
        if (l_iter->precision < f_precisionThreshold ||
            l_iter->lambda2 < l_Tl2 ||
            l_iter->x < l_sigma ||
            l_iter->y < l_sigma ||
            l_iter->x > f_width - l_sigma ||
            l_iter->y > f_height - l_sigma) {
            l_iter = m_features_p->erase(l_iter);
        }
        else {
            l_iter++;
        }
    }
}

void CSfop::nonMaximaSuppression(
        const unsigned int f_numLayers)
{
    // sort features by precision
    m_features_p->sort(sortByPrecision);

    // loop through all features in set
    for (std::list<CFeature>::iterator
            it1 = m_features_p->begin();
            it1 != m_features_p->end(); ++it1) {

        // loop through remaining features with smaller precision
        std::list<CFeature>::iterator it2 = ++it1;
        for (--it1; it2 != m_features_p->end(); ++it2) {

            // compute Mahalanobis distance (distance in x, y, sigma)
            const float l_dx = it1->x - it2->x;
            const float l_dy = it1->y - it2->y;
            const float l_meanSigmaSqr =
                it1->sigma * it1->sigma + it2->sigma * it2->sigma;
            const float l_scaleTerm = 0.5f * f_numLayers *
                std::log(it1->sigma / it2->sigma) / 0.693147181; // 0.69... = ln(2)
            const float l_distSqr =
                l_dx * l_dx / l_meanSigmaSqr +
                l_dy * l_dy / l_meanSigmaSqr +
                l_scaleTerm * l_scaleTerm;

            // remove feature if closer than threshold
            if (l_distSqr < 1.0f) {
                it2 = m_features_p->erase(it2);
                it2--;
            }

        }
    }
}

void CSfop::writeFile(const char f_filename[]) const
{
    // open file
    std::ofstream l_datFile;
    l_datFile.open(f_filename);

    // write header
    l_datFile << "1.0" << std::endl;
    l_datFile << m_features_p->size() << std::endl;

    // write features
    for (std::list<CFeature>::const_iterator
            l_iter = m_features_p->begin();
            l_iter != m_features_p->end(); l_iter++) {
        l_datFile
            << l_iter->x << " "
            << l_iter->y << " "
            << std::pow(l_iter->sigma, -2.0f) << " "
            << 0.0f << " "
            << std::pow(l_iter->sigma, -2.0f) << std::endl;
    }

    // close file
    l_datFile.close();
}

}

