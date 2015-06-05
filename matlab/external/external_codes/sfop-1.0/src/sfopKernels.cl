__kernel void downsample(
    const __global float* A,
    const int W,
    const int H,
    const int P,
    __global float* C)
{
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    C[X + Y * P] = A[min(2 * X, W - 1) + min(2 * Y, H - 1) * W];
}

__kernel void findLocalMax(
    const __global float* lowerLayer_p,
    const __global float* middleLayer_p,
    const __global float* upperLayer_p,
    __global float4* result,
    const int W,
    const int H,
    __local float* lowerCache,
    __local float* middleCache,
    __local float* upperCache,
    const int w,
    const int h)
{
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    if (X < 0 || X >= W || Y < 0 || Y >= H) return;
    const int x = get_local_id(0);
    const int y = get_local_id(1);

    // copy a block of each layer with one pixel frame to register memory
    const int Idx = X + Y * W;
    const int idx = x + 1 + (y + 1) * (w + 2);
    lowerCache[ idx] = lowerLayer_p[ Idx];
    middleCache[idx] = middleLayer_p[Idx];
    upperCache[ idx] = upperLayer_p[ Idx];
    const float p = middleCache[idx];
    if (p == NAN || p <= 0) {
        result[Idx].w = 0.0f;
        return;
    }
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int dx = i - 1;
            const int dy = j - 1;
            if (x + dx < 0 || x + dx >= w ||
                y + dy < 0 || y + dy >= h ||
                (dx == 0 && dy == 0)) {
                const int idx_ = x + i + (y + j) * (w + 2);
                const int Idx_ = min(max(X + dx, 0), W - 1) + min(max(Y + dy, 0), H - 1) * (int) W;
                lowerCache[ idx_] = lowerLayer_p[ Idx_];
                middleCache[idx_] = middleLayer_p[Idx_];
                upperCache[ idx_] = upperLayer_p[ Idx_];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // check neighborhood
    #pragma unroll
    for (int i = 0; i < 3; ++i) { 
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int idx_ = x + i + (y + j) * (w + 2);
            if (p < lowerCache[ idx_] ||
                p < middleCache[idx_] ||
                p < upperCache[ idx_]) {
                result[Idx].w = 0.0f;
                return;
            }
        }
    }

    // copy neighborhood
    float P[27];
    #pragma unroll
    for (int i = 0; i < 3; ++i) { 
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int idx_ = x + i + (y + j) * (w + 2);
            P[i + j * 3        ] = lowerCache[ idx_];
            P[i + j * 3 +     9] = middleCache[idx_];
            P[i + j * 3 + 2 * 9] = upperCache[ idx_];
        }
    }

    // calculate Hessian matrix
    const float HM[9] = {
        + P[ 0] / 16.0f - P[ 1] /  8.0f + P[ 2] / 16.0f
        + P[ 3] /  8.0f - P[ 4] /  4.0f + P[ 5] /  8.0f
        + P[ 6] / 16.0f - P[ 7] /  8.0f + P[ 8] / 16.0f
        + P[ 9] /  8.0f - P[10] /  4.0f + P[11] /  8.0f
        + P[12] /  4.0f - P[13] /  2.0f + P[14] /  4.0f
        + P[15] /  8.0f - P[16] /  4.0f + P[17] /  8.0f
        + P[18] / 16.0f - P[19] /  8.0f + P[20] / 16.0f
        + P[21] /  8.0f - P[22] /  4.0f + P[23] /  8.0f
        + P[24] / 16.0f - P[25] /  8.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f - P[ 2] / 16.0f - P[ 6] / 16.0f
        + P[ 8] / 16.0f + P[ 9] /  8.0f - P[11] /  8.0f
        - P[15] /  8.0f + P[17] /  8.0f + P[18] / 16.0f
        - P[20] / 16.0f - P[24] / 16.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f - P[ 2] / 16.0f + P[ 3] /  8.0f
        - P[ 5] /  8.0f + P[ 6] / 16.0f - P[ 8] / 16.0f
        - P[18] / 16.0f + P[20] / 16.0f - P[21] /  8.0f
        + P[23] /  8.0f - P[24] / 16.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f - P[ 2] / 16.0f - P[ 6] / 16.0f
        + P[ 8] / 16.0f + P[ 9] /  8.0f - P[11] /  8.0f
        - P[15] /  8.0f + P[17] /  8.0f + P[18] / 16.0f
        - P[20] / 16.0f - P[24] / 16.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
        - P[ 3] /  8.0f - P[ 4] /  4.0f - P[ 5] /  8.0f
        + P[ 6] / 16.0f + P[ 7] /  8.0f + P[ 8] / 16.0f
        + P[ 9] /  8.0f + P[10] /  4.0f + P[11] /  8.0f
        - P[12] /  4.0f - P[13] /  2.0f - P[14] /  4.0f
        + P[15] /  8.0f + P[16] /  4.0f + P[17] /  8.0f
        + P[18] / 16.0f + P[19] /  8.0f + P[20] / 16.0f
        - P[21] /  8.0f - P[22] /  4.0f - P[23] /  8.0f
        + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
        - P[ 6] / 16.0f - P[ 7] /  8.0f - P[ 8] / 16.0f
        - P[18] / 16.0f - P[19] /  8.0f - P[20] / 16.0f
        + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f - P[ 2] / 16.0f + P[ 3] /  8.0f
        - P[ 5] /  8.0f + P[ 6] / 16.0f - P[ 8] / 16.0f
        - P[18] / 16.0f + P[20] / 16.0f - P[21] /  8.0f
        + P[23] /  8.0f - P[24] / 16.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
        - P[ 6] / 16.0f - P[ 7] /  8.0f - P[ 8] / 16.0f
        - P[18] / 16.0f - P[19] /  8.0f - P[20] / 16.0f
        + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f,
        + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
        + P[ 3] /  8.0f + P[ 4] /  4.0f + P[ 5] /  8.0f
        + P[ 6] / 16.0f + P[ 7] /  8.0f + P[ 8] / 16.0f
        - P[ 9] /  8.0f - P[10] /  4.0f - P[11] /  8.0f
        - P[12] /  4.0f - P[13] /  2.0f - P[14] /  4.0f
        - P[15] /  8.0f - P[16] /  4.0f - P[17] /  8.0f
        + P[18] / 16.0f + P[19] /  8.0f + P[20] / 16.0f
        + P[21] /  8.0f + P[22] /  4.0f + P[23] /  8.0f
        + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f};

    // check for negative definiteness
    const float det =
        HM[0] * (HM[4] * HM[8] - HM[5] * HM[7]) -
        HM[3] * (HM[1] * HM[8] - HM[2] * HM[7]) +
        HM[6] * (HM[1] * HM[5] - HM[2] * HM[4]);
    if (HM[0] < 0.0f &&
        HM[0] * HM[4] - HM[1] * HM[3] > 0.0f &&
        det < 0.0f) {
    }
    else {
        result[Idx].w = 0.0f;
        return;
    }

    // calculate the inverse
    const float HI[9] = {
        (HM[4] * HM[8] - HM[5] * HM[7]) / det,
        (HM[2] * HM[7] - HM[1] * HM[8]) / det,
        (HM[1] * HM[5] - HM[2] * HM[4]) / det,
        (HM[5] * HM[6] - HM[3] * HM[8]) / det,
        (HM[0] * HM[8] - HM[2] * HM[6]) / det,
        (HM[2] * HM[3] - HM[0] * HM[5]) / det,
        (HM[3] * HM[7] - HM[4] * HM[6]) / det,
        (HM[1] * HM[6] - HM[0] * HM[7]) / det,
        (HM[0] * HM[4] - HM[1] * HM[3]) / det};

    // calculate gradient
    const float g[3] = {
        - P[ 0] / 32.0f + P[ 2] / 32.0f - P[ 3] / 16.0f
        + P[ 5] / 16.0f - P[ 6] / 32.0f + P[ 8] / 32.0f
        - P[ 9] / 16.0f + P[11] / 16.0f - P[12] /  8.0f
        + P[14] /  8.0f - P[15] / 16.0f + P[17] / 16.0f
        - P[18] / 32.0f + P[20] / 32.0f - P[21] / 16.0f
        + P[23] / 16.0f - P[24] / 32.0f + P[26] / 32.0f,
        - P[ 0] / 32.0f - P[ 1] / 16.0f - P[ 2] / 32.0f
        + P[ 6] / 32.0f + P[ 7] / 16.0f + P[ 8] / 32.0f
        - P[ 9] / 16.0f - P[10] /  8.0f - P[11] / 16.0f
        + P[15] / 16.0f + P[16] /  8.0f + P[17] / 16.0f
        - P[18] / 32.0f - P[19] / 16.0f - P[20] / 32.0f
        + P[24] / 32.0f + P[25] / 16.0f + P[26] / 32.0f,
        - P[ 0] / 32.0f - P[ 1] / 16.0f - P[ 2] / 32.0f
        - P[ 3] / 16.0f - P[ 4] /  8.0f - P[ 5] / 16.0f
        - P[ 6] / 32.0f - P[ 7] / 16.0f - P[ 8] / 32.0f
        + P[18] / 32.0f + P[19] / 16.0f + P[20] / 32.0f
        + P[21] / 16.0f + P[22] /  8.0f + P[23] / 16.0f
        + P[24] / 32.0f + P[25] / 16.0f + P[26] / 32.0f};
    
    // compute update = - H^-1 * g
    float update[3];
    #pragma unroll
    for (int y = 0; y < 3; ++y) {
        update[y] = 0.0f;
        #pragma unroll
        for (int x = 0; x < 3; ++x) {
            update[y] = update[y] - g[x] * HI[x + 3 * y];
        }
    }

    // check for divergence
    if (update[0] * update[0] +
        update[1] * update[1] +
        update[2] * update[2] > 1.0f) {
        result[Idx].w = 0.0f;
        return;
    }

    // return result
    result[Idx].x = X + update[0];
    result[Idx].y = Y + update[1];
    result[Idx].z =     update[2];
    result[Idx].w = P[13] + 0.5f * (g[0] * update[0] + g[1] * update[1] + g[2] * update[2]);
}

#define ROWS_BLOCKDIM_X      32
#define ROWS_BLOCKDIM_Y       4
#define ROWS_RESULT_STEPS     8
#define ROWS_HALO_STEPS       1
__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1)))
void convRow(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius)
{
    __local float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    // memory offsets to the left halo edge
    const int baseX = (get_group_id(0) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + get_local_id(0);
    const int baseY = get_group_id(1) * ROWS_BLOCKDIM_Y + get_local_id(1);
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    // main data
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    // left halo
    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = baseX + i * ROWS_BLOCKDIM_X >= 0 ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[-baseX];
    }

    // right halo
    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = baseX + i * ROWS_BLOCKDIM_X < imageW ? d_Src[i * ROWS_BLOCKDIM_X] : d_Src[imageW - 1 - baseX];
    }

    // result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;
        for (int j = -kernel_radius; j <= kernel_radius; j++) {
            sum += c_Kernel[kernel_radius - j] * l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X + j];
        }
        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

#define COLUMNS_BLOCKDIM_X   32
#define COLUMNS_BLOCKDIM_Y    8
#define COLUMNS_RESULT_STEPS  8
#define COLUMNS_HALO_STEPS    2
__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void convCol(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius)
{
    __local float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    // memory offsets to the upper halo edge
    const int baseX = get_group_id(0) * COLUMNS_BLOCKDIM_X + get_local_id(0);
    const int baseY = (get_group_id(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + get_local_id(1);
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    // main data
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    // upper halo
    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = baseY + i * COLUMNS_BLOCKDIM_Y >= 0 ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : d_Src[-baseY * pitch];
    }

    // lower halo
    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y]  = baseY + i * COLUMNS_BLOCKDIM_Y < imageH ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : d_Src[(imageH - 1 - baseY) * pitch];
    }

    // result
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;
        for (int j = -kernel_radius; j <= kernel_radius; j++) {
            sum += c_Kernel[kernel_radius - j] * l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y + j];
        }
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

__kernel void hessian(
    const __global float* P,
    __global float* HM)
{
    HM[0] = + P[ 0] / 16.0f - P[ 1] /  8.0f + P[ 2] / 16.0f
            + P[ 3] /  8.0f - P[ 4] /  4.0f + P[ 5] /  8.0f
            + P[ 6] / 16.0f - P[ 7] /  8.0f + P[ 8] / 16.0f
            + P[ 9] /  8.0f - P[10] /  4.0f + P[11] /  8.0f
            + P[12] /  4.0f - P[13] /  2.0f + P[14] /  4.0f
            + P[15] /  8.0f - P[16] /  4.0f + P[17] /  8.0f
            + P[18] / 16.0f - P[19] /  8.0f + P[20] / 16.0f
            + P[21] /  8.0f - P[22] /  4.0f + P[23] /  8.0f
            + P[24] / 16.0f - P[25] /  8.0f + P[26] / 16.0f;
    HM[1] = + P[ 0] / 16.0f - P[ 2] / 16.0f - P[ 6] / 16.0f
            + P[ 8] / 16.0f + P[ 9] /  8.0f - P[11] /  8.0f
            - P[15] /  8.0f + P[17] /  8.0f + P[18] / 16.0f
            - P[20] / 16.0f - P[24] / 16.0f + P[26] / 16.0f;
    HM[2] = + P[ 0] / 16.0f - P[ 2] / 16.0f + P[ 3] /  8.0f
            - P[ 5] /  8.0f + P[ 6] / 16.0f - P[ 8] / 16.0f
            - P[18] / 16.0f + P[20] / 16.0f - P[21] /  8.0f
            + P[23] /  8.0f - P[24] / 16.0f + P[26] / 16.0f;
    HM[3] = + P[ 0] / 16.0f - P[ 2] / 16.0f - P[ 6] / 16.0f
            + P[ 8] / 16.0f + P[ 9] /  8.0f - P[11] /  8.0f
            - P[15] /  8.0f + P[17] /  8.0f + P[18] / 16.0f
            - P[20] / 16.0f - P[24] / 16.0f + P[26] / 16.0f;
    HM[4] = + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
            - P[ 3] /  8.0f - P[ 4] /  4.0f - P[ 5] /  8.0f
            + P[ 6] / 16.0f + P[ 7] /  8.0f + P[ 8] / 16.0f
            + P[ 9] /  8.0f + P[10] /  4.0f + P[11] /  8.0f
            - P[12] /  4.0f - P[13] /  2.0f - P[14] /  4.0f
            + P[15] /  8.0f + P[16] /  4.0f + P[17] /  8.0f
            + P[18] / 16.0f + P[19] /  8.0f + P[20] / 16.0f
            - P[21] /  8.0f - P[22] /  4.0f - P[23] /  8.0f
            + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f;
    HM[5] = + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
            - P[ 6] / 16.0f - P[ 7] /  8.0f - P[ 8] / 16.0f
            - P[18] / 16.0f - P[19] /  8.0f - P[20] / 16.0f
            + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f;
    HM[6] = + P[ 0] / 16.0f - P[ 2] / 16.0f + P[ 3] /  8.0f
            - P[ 5] /  8.0f + P[ 6] / 16.0f - P[ 8] / 16.0f
            - P[18] / 16.0f + P[20] / 16.0f - P[21] /  8.0f
            + P[23] /  8.0f - P[24] / 16.0f + P[26] / 16.0f;
    HM[7] = + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
            - P[ 6] / 16.0f - P[ 7] /  8.0f - P[ 8] / 16.0f
            - P[18] / 16.0f - P[19] /  8.0f - P[20] / 16.0f
            + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f;
    HM[8] = + P[ 0] / 16.0f + P[ 1] /  8.0f + P[ 2] / 16.0f
            + P[ 3] /  8.0f + P[ 4] /  4.0f + P[ 5] /  8.0f
            + P[ 6] / 16.0f + P[ 7] /  8.0f + P[ 8] / 16.0f
            - P[ 9] /  8.0f - P[10] /  4.0f - P[11] /  8.0f
            - P[12] /  4.0f - P[13] /  2.0f - P[14] /  4.0f
            - P[15] /  8.0f - P[16] /  4.0f - P[17] /  8.0f
            + P[18] / 16.0f + P[19] /  8.0f + P[20] / 16.0f
            + P[21] /  8.0f + P[22] /  4.0f + P[23] /  8.0f
            + P[24] / 16.0f + P[25] /  8.0f + P[26] / 16.0f;
}

__kernel void inverse(
    const __global float* HM,
    __global float* HI)
{
    const float det =
        HM[0] * (HM[4] * HM[8] - HM[5] * HM[7]) -
        HM[3] * (HM[1] * HM[8] - HM[2] * HM[7]) +
        HM[6] * (HM[1] * HM[5] - HM[2] * HM[4]);
    if (det == 0.0f) return;
    HI[0] = (HM[4] * HM[8] - HM[5] * HM[7]) / det;
    HI[1] = (HM[2] * HM[7] - HM[1] * HM[8]) / det;
    HI[2] = (HM[1] * HM[5] - HM[2] * HM[4]) / det;
    HI[3] = (HM[5] * HM[6] - HM[3] * HM[8]) / det;
    HI[4] = (HM[0] * HM[8] - HM[2] * HM[6]) / det;
    HI[5] = (HM[2] * HM[3] - HM[0] * HM[5]) / det;
    HI[6] = (HM[3] * HM[7] - HM[4] * HM[6]) / det;
    HI[7] = (HM[1] * HM[6] - HM[0] * HM[7]) / det;
    HI[8] = (HM[0] * HM[4] - HM[1] * HM[3]) / det;
}

__kernel void gradient(
    const __global float* P,
    __global float* g)
{
    g[0] = - P[ 0] / 32.0f + P[ 2] / 32.0f - P[ 3] / 16.0f
           + P[ 5] / 16.0f - P[ 6] / 32.0f + P[ 8] / 32.0f
           - P[ 9] / 16.0f + P[11] / 16.0f - P[12] /  8.0f
           + P[14] /  8.0f - P[15] / 16.0f + P[17] / 16.0f
           - P[18] / 32.0f + P[20] / 32.0f - P[21] / 16.0f
           + P[23] / 16.0f - P[24] / 32.0f + P[26] / 32.0f;
    g[1] = - P[ 0] / 32.0f - P[ 1] / 16.0f - P[ 2] / 32.0f
           + P[ 6] / 32.0f + P[ 7] / 16.0f + P[ 8] / 32.0f
           - P[ 9] / 16.0f - P[10] /  8.0f - P[11] / 16.0f
           + P[15] / 16.0f + P[16] /  8.0f + P[17] / 16.0f
           - P[18] / 32.0f - P[19] / 16.0f - P[20] / 32.0f
           + P[24] / 32.0f + P[25] / 16.0f + P[26] / 32.0f;
    g[2] = - P[ 0] / 32.0f - P[ 1] / 16.0f - P[ 2] / 32.0f
           - P[ 3] / 16.0f - P[ 4] /  8.0f - P[ 5] / 16.0f
           - P[ 6] / 32.0f - P[ 7] / 16.0f - P[ 8] / 32.0f
           + P[18] / 32.0f + P[19] / 16.0f + P[20] / 32.0f
           + P[21] / 16.0f + P[22] /  8.0f + P[23] / 16.0f
           + P[24] / 32.0f + P[25] / 16.0f + P[26] / 32.0f;
}

__kernel void solver(
    const __global float* HM,
    const __global float* g,
    __global float* update)
{
    const float det =
        HM[0] * (HM[4] * HM[8] - HM[5] * HM[7]) -
        HM[3] * (HM[1] * HM[8] - HM[2] * HM[7]) +
        HM[6] * (HM[1] * HM[5] - HM[2] * HM[4]);
    if (det == 0.0f) return;

    const float HI[9] = {
        (HM[4] * HM[8] - HM[5] * HM[7]) / det,
        (HM[2] * HM[7] - HM[1] * HM[8]) / det,
        (HM[1] * HM[5] - HM[2] * HM[4]) / det,
        (HM[5] * HM[6] - HM[3] * HM[8]) / det,
        (HM[0] * HM[8] - HM[2] * HM[6]) / det,
        (HM[2] * HM[3] - HM[0] * HM[5]) / det,
        (HM[3] * HM[7] - HM[4] * HM[6]) / det,
        (HM[1] * HM[6] - HM[0] * HM[7]) / det,
        (HM[0] * HM[4] - HM[1] * HM[3]) / det};
    
    #pragma unroll
    for (int y = 0; y < 3; ++y) {
        update[y] = 0.0f;
        #pragma unroll
        for (int x = 0; x < 3; ++x) {
            update[y] = update[y] - g[x] * HI[x + 3 * y];
        }
    }
}

__kernel void negDefinite(
    const __global float* HM,
    __global float* isNegDefinite)
{
    const float det =
        HM[0] * (HM[4] * HM[8] - HM[5] * HM[7]) -
        HM[3] * (HM[1] * HM[8] - HM[2] * HM[7]) +
        HM[6] * (HM[1] * HM[5] - HM[2] * HM[4]);
    const bool ans =
        (HM[0] < 0.0f) &&
        (HM[0] * HM[4] - HM[1] * HM[3] > 0.0f) &&
        (det < 0.0f);
    isNegDefinite[0] = ans ? 1.0f : 0.0f;
}

__kernel void filter(
    const unsigned short int filterName,
    const float sigma,
    const unsigned short int kSize,
    __global float* F)
{
    const int X = get_global_id(0);
    const float dX = X - kSize;
    const float f = 0.398942280401433 / sigma;
    switch (filterName) {
        case 0:
            F[X] = f * exp(-0.5f * dX * dX / sigma / sigma);
            break;
        case 1:
            F[X] = f * 0.5f * (
                exp(-0.5f * (dX + 1.0f) * (dX + 1.0f) / sigma / sigma) -
                exp(-0.5f * (dX - 1.0f) * (dX - 1.0f) / sigma / sigma));
            break;
        case 2:
            F[X] = f * exp(-0.5f * dX * dX / sigma / sigma) * dX;
            break;
        case 3:
            F[X] = f * exp(-0.5f * dX * dX / sigma / sigma) * dX * dX;
            break;
    }
}

__kernel void triSqr(
    const __global float* gx,
    const __global float* gy,
    __global float* gx2,
    __global float* gxy,
    __global float* gy2)
{
    const int X = get_global_id(0);
    const float l_gx = gx[X];
    const float l_gy = gy[X];
    gx2[X] = l_gx * l_gx;
    gxy[X] = l_gx * l_gy;
    gy2[X] = l_gy * l_gy;
}

__kernel void lambda2(
    const float M,
    const __global float* Nxx,
    const __global float* Nxy,
    const __global float* Nyy,
    __global float* lambda2)
{
    const int X = get_global_id(0);
    const float l_Nxx = Nxx[X];
    const float l_Nxy = Nxy[X];
    const float l_Nyy = Nyy[X];
    const float traceHalf = 0.5f * (l_Nxx + l_Nyy);
    const float det = l_Nxx * l_Nyy - l_Nxy * l_Nxy;
    lambda2[X] = (traceHalf - sqrt(traceHalf * traceHalf - det)) * M;
}

__constant float cosLUT[360] = {
 1.000000000,  0.999847695,  0.999390827,  0.998629535,  0.997564050,  0.996194698,  0.994521895,  0.992546152,  0.990268069,  0.987688341,
 0.984807753,  0.981627183,  0.978147601,  0.974370065,  0.970295726,  0.965925826,  0.961261696,  0.956304756,  0.951056516,  0.945518576,
 0.939692621,  0.933580426,  0.927183855,  0.920504853,  0.913545458,  0.906307787,  0.898794046,  0.891006524,  0.882947593,  0.874619707,
 0.866025404,  0.857167301,  0.848048096,  0.838670568,  0.829037573,  0.819152044,  0.809016994,  0.798635510,  0.788010754,  0.777145961,
 0.766044443,  0.754709580,  0.743144825,  0.731353702,  0.719339800,  0.707106781,  0.694658370,  0.681998360,  0.669130606,  0.656059029,
 0.642787610,  0.629320391,  0.615661475,  0.601815023,  0.587785252,  0.573576436,  0.559192903,  0.544639035,  0.529919264,  0.515038075,
 0.500000000,  0.484809620,  0.469471563,  0.453990500,  0.438371147,  0.422618262,  0.406736643,  0.390731128,  0.374606593,  0.358367950,
 0.342020143,  0.325568154,  0.309016994,  0.292371705,  0.275637356,  0.258819045,  0.241921896,  0.224951054,  0.207911691,  0.190808995,
 0.173648178,  0.156434465,  0.139173101,  0.121869343,  0.104528463,  0.087155743,  0.069756474,  0.052335956,  0.034899497,  0.017452406,
-0.000000000, -0.017452406, -0.034899497, -0.052335956, -0.069756474, -0.087155743, -0.104528463, -0.121869343, -0.139173101, -0.156434465,
-0.173648178, -0.190808995, -0.207911691, -0.224951054, -0.241921896, -0.258819045, -0.275637356, -0.292371705, -0.309016994, -0.325568154,
-0.342020143, -0.358367950, -0.374606593, -0.390731128, -0.406736643, -0.422618262, -0.438371147, -0.453990500, -0.469471563, -0.484809620,
-0.500000000, -0.515038075, -0.529919264, -0.544639035, -0.559192903, -0.573576436, -0.587785252, -0.601815023, -0.615661475, -0.629320391,
-0.642787610, -0.656059029, -0.669130606, -0.681998360, -0.694658370, -0.707106781, -0.719339800, -0.731353702, -0.743144825, -0.754709580,
-0.766044443, -0.777145961, -0.788010754, -0.798635510, -0.809016994, -0.819152044, -0.829037573, -0.838670568, -0.848048096, -0.857167301,
-0.866025404, -0.874619707, -0.882947593, -0.891006524, -0.898794046, -0.906307787, -0.913545458, -0.920504853, -0.927183855, -0.933580426,
-0.939692621, -0.945518576, -0.951056516, -0.956304756, -0.961261696, -0.965925826, -0.970295726, -0.974370065, -0.978147601, -0.981627183,
-0.984807753, -0.987688341, -0.990268069, -0.992546152, -0.994521895, -0.996194698, -0.997564050, -0.998629535, -0.999390827, -0.999847695,
-1.000000000, -0.999847695, -0.999390827, -0.998629535, -0.997564050, -0.996194698, -0.994521895, -0.992546152, -0.990268069, -0.987688341,
-0.984807753, -0.981627183, -0.978147601, -0.974370065, -0.970295726, -0.965925826, -0.961261696, -0.956304756, -0.951056516, -0.945518576,
-0.939692621, -0.933580426, -0.927183855, -0.920504853, -0.913545458, -0.906307787, -0.898794046, -0.891006524, -0.882947593, -0.874619707,
-0.866025404, -0.857167301, -0.848048096, -0.838670568, -0.829037573, -0.819152044, -0.809016994, -0.798635510, -0.788010754, -0.777145961,
-0.766044443, -0.754709580, -0.743144825, -0.731353702, -0.719339800, -0.707106781, -0.694658370, -0.681998360, -0.669130606, -0.656059029,
-0.642787610, -0.629320391, -0.615661475, -0.601815023, -0.587785252, -0.573576436, -0.559192903, -0.544639035, -0.529919264, -0.515038075,
-0.500000000, -0.484809620, -0.469471563, -0.453990500, -0.438371147, -0.422618262, -0.406736643, -0.390731128, -0.374606593, -0.358367950,
-0.342020143, -0.325568154, -0.309016994, -0.292371705, -0.275637356, -0.258819045, -0.241921896, -0.224951054, -0.207911691, -0.190808995,
-0.173648178, -0.156434465, -0.139173101, -0.121869343, -0.104528463, -0.087155743, -0.069756474, -0.052335956, -0.034899497, -0.017452406,
 0.000000000,  0.017452406,  0.034899497,  0.052335956,  0.069756474,  0.087155743,  0.104528463,  0.121869343,  0.139173101,  0.156434465,
 0.173648178,  0.190808995,  0.207911691,  0.224951054,  0.241921896,  0.258819045,  0.275637356,  0.292371705,  0.309016994,  0.325568154,
 0.342020143,  0.358367950,  0.374606593,  0.390731128,  0.406736643,  0.422618262,  0.438371147,  0.453990500,  0.469471563,  0.484809620,
 0.500000000,  0.515038075,  0.529919264,  0.544639035,  0.559192903,  0.573576436,  0.587785252,  0.601815023,  0.615661475,  0.629320391,
 0.642787610,  0.656059029,  0.669130606,  0.681998360,  0.694658370,  0.707106781,  0.719339800,  0.731353702,  0.743144825,  0.754709580,
 0.766044443,  0.777145961,  0.788010754,  0.798635510,  0.809016994,  0.819152044,  0.829037573,  0.838670568,  0.848048096,  0.857167301,
 0.866025404,  0.874619707,  0.882947593,  0.891006524,  0.898794046,  0.906307787,  0.913545458,  0.920504853,  0.927183855,  0.933580426,
 0.939692621,  0.945518576,  0.951056516,  0.956304756,  0.961261696,  0.965925826,  0.970295726,  0.974370065,  0.978147601,  0.981627183,
 0.984807753,  0.987688341,  0.990268069,  0.992546152,  0.994521895,  0.996194698,  0.997564050,  0.998629535,  0.999390827,  0.999847695};

__constant float sinLUT[360] = {
 0.000000000,  0.017452406,  0.034899497,  0.052335956,  0.069756474,  0.087155743,  0.104528463,  0.121869343,  0.139173101,  0.156434465,
 0.173648178,  0.190808995,  0.207911691,  0.224951054,  0.241921896,  0.258819045,  0.275637356,  0.292371705,  0.309016994,  0.325568154,
 0.342020143,  0.358367950,  0.374606593,  0.390731128,  0.406736643,  0.422618262,  0.438371147,  0.453990500,  0.469471563,  0.484809620,
 0.500000000,  0.515038075,  0.529919264,  0.544639035,  0.559192903,  0.573576436,  0.587785252,  0.601815023,  0.615661475,  0.629320391,
 0.642787610,  0.656059029,  0.669130606,  0.681998360,  0.694658370,  0.707106781,  0.719339800,  0.731353702,  0.743144825,  0.754709580,
 0.766044443,  0.777145961,  0.788010754,  0.798635510,  0.809016994,  0.819152044,  0.829037573,  0.838670568,  0.848048096,  0.857167301,
 0.866025404,  0.874619707,  0.882947593,  0.891006524,  0.898794046,  0.906307787,  0.913545458,  0.920504853,  0.927183855,  0.933580426,
 0.939692621,  0.945518576,  0.951056516,  0.956304756,  0.961261696,  0.965925826,  0.970295726,  0.974370065,  0.978147601,  0.981627183,
 0.984807753,  0.987688341,  0.990268069,  0.992546152,  0.994521895,  0.996194698,  0.997564050,  0.998629535,  0.999390827,  0.999847695,
 1.000000000,  0.999847695,  0.999390827,  0.998629535,  0.997564050,  0.996194698,  0.994521895,  0.992546152,  0.990268069,  0.987688341,
 0.984807753,  0.981627183,  0.978147601,  0.974370065,  0.970295726,  0.965925826,  0.961261696,  0.956304756,  0.951056516,  0.945518576,
 0.939692621,  0.933580426,  0.927183855,  0.920504853,  0.913545458,  0.906307787,  0.898794046,  0.891006524,  0.882947593,  0.874619707,
 0.866025404,  0.857167301,  0.848048096,  0.838670568,  0.829037573,  0.819152044,  0.809016994,  0.798635510,  0.788010754,  0.777145961,
 0.766044443,  0.754709580,  0.743144825,  0.731353702,  0.719339800,  0.707106781,  0.694658370,  0.681998360,  0.669130606,  0.656059029,
 0.642787610,  0.629320391,  0.615661475,  0.601815023,  0.587785252,  0.573576436,  0.559192903,  0.544639035,  0.529919264,  0.515038075,
 0.500000000,  0.484809620,  0.469471563,  0.453990500,  0.438371147,  0.422618262,  0.406736643,  0.390731128,  0.374606593,  0.358367950,
 0.342020143,  0.325568154,  0.309016994,  0.292371705,  0.275637356,  0.258819045,  0.241921896,  0.224951054,  0.207911691,  0.190808995,
 0.173648178,  0.156434465,  0.139173101,  0.121869343,  0.104528463,  0.087155743,  0.069756474,  0.052335956,  0.034899497,  0.017452406,
-0.000000000, -0.017452406, -0.034899497, -0.052335956, -0.069756474, -0.087155743, -0.104528463, -0.121869343, -0.139173101, -0.156434465,
-0.173648178, -0.190808995, -0.207911691, -0.224951054, -0.241921896, -0.258819045, -0.275637356, -0.292371705, -0.309016994, -0.325568154,
-0.342020143, -0.358367950, -0.374606593, -0.390731128, -0.406736643, -0.422618262, -0.438371147, -0.453990500, -0.469471563, -0.484809620,
-0.500000000, -0.515038075, -0.529919264, -0.544639035, -0.559192903, -0.573576436, -0.587785252, -0.601815023, -0.615661475, -0.629320391,
-0.642787610, -0.656059029, -0.669130606, -0.681998360, -0.694658370, -0.707106781, -0.719339800, -0.731353702, -0.743144825, -0.754709580,
-0.766044443, -0.777145961, -0.788010754, -0.798635510, -0.809016994, -0.819152044, -0.829037573, -0.838670568, -0.848048096, -0.857167301,
-0.866025404, -0.874619707, -0.882947593, -0.891006524, -0.898794046, -0.906307787, -0.913545458, -0.920504853, -0.927183855, -0.933580426,
-0.939692621, -0.945518576, -0.951056516, -0.956304756, -0.961261696, -0.965925826, -0.970295726, -0.974370065, -0.978147601, -0.981627183,
-0.984807753, -0.987688341, -0.990268069, -0.992546152, -0.994521895, -0.996194698, -0.997564050, -0.998629535, -0.999390827, -0.999847695,
-1.000000000, -0.999847695, -0.999390827, -0.998629535, -0.997564050, -0.996194698, -0.994521895, -0.992546152, -0.990268069, -0.987688341,
-0.984807753, -0.981627183, -0.978147601, -0.974370065, -0.970295726, -0.965925826, -0.961261696, -0.956304756, -0.951056516, -0.945518576,
-0.939692621, -0.933580426, -0.927183855, -0.920504853, -0.913545458, -0.906307787, -0.898794046, -0.891006524, -0.882947593, -0.874619707,
-0.866025404, -0.857167301, -0.848048096, -0.838670568, -0.829037573, -0.819152044, -0.809016994, -0.798635510, -0.788010754, -0.777145961,
-0.766044443, -0.754709580, -0.743144825, -0.731353702, -0.719339800, -0.707106781, -0.694658370, -0.681998360, -0.669130606, -0.656059029,
-0.642787610, -0.629320391, -0.615661475, -0.601815023, -0.587785252, -0.573576436, -0.559192903, -0.544639035, -0.529919264, -0.515038075,
-0.500000000, -0.484809620, -0.469471563, -0.453990500, -0.438371147, -0.422618262, -0.406736643, -0.390731128, -0.374606593, -0.358367950,
-0.342020143, -0.325568154, -0.309016994, -0.292371705, -0.275637356, -0.258819045, -0.241921896, -0.224951054, -0.207911691, -0.190808995,
-0.173648178, -0.156434465, -0.139173101, -0.121869343, -0.104528463, -0.087155743, -0.069756474, -0.052335956, -0.034899497, -0.017452406};

__kernel void triSqrAlpha(
    const float alpha,
    const __global float* gx,
    const __global float* gy,
    __global float* gx2a,
    __global float* gxyaTwice,
    __global float* gy2a)
{
    const int X = get_global_id(0);
    const float l_gx = gx[X];
    const float l_gy = gy[X];
    const float l_cosa = cosLUT[(int) (alpha * 57.29578f + 0.5f)];
    const float l_sina = sinLUT[(int) (alpha * 57.29578f + 0.5f)];
    const float l_gxa =   l_gx * l_cosa + l_gy * l_sina;
    const float l_gya = - l_gx * l_sina + l_gy * l_cosa;
    const float l_gxya = l_gxa * l_gya;
    gx2a[X]      = l_gxa * l_gxa;
    gxyaTwice[X] = l_gxya + l_gxya;
    gy2a[X]      = l_gya * l_gya;
}

__kernel void triSum(
    const __global float* a,
    const __global float* b,
    const __global float* c,
    __global float* sum)
{
    const int X = get_global_id(0);
    sum[X] = a[X] + b[X] + c[X];
}

__kernel void precision(
    const float factor,
    const __global float* lambda2,
    const __global float* omega,
    __global float* precision)
{
    const int X = get_global_id(0);
    precision[X] = factor * lambda2[X] / omega[X];
}

__kernel void bestOmega(
    const __global float* omega0,
    const __global float* omega60,
    const __global float* omega120,
    __global float* omegaMin)
{
    const int X = get_global_id(0);
    const float om0   = omega0[X];
    const float om60  = omega60[X];
    const float om120 = omega120[X];
    omegaMin[X] = (
            om0 + om60 + om120 -
            2.0f * sqrt(
                om0 * om0  + om60 * om60  + om120 * om120 -
                om0 * om60 - om60 * om120 - om120 * om0)
            ) / 3.0f;
}

