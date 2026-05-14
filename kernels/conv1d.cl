__kernel void conv1d(
    const int stride, 
    const int dilation,
    const int padding,
    const int C_in,
    const int L_in,
    const int K_size,
    
    __constant float *x,
    __constant float *weight,
    __global float *output)
{
    const int n = get_global_id(0);
    const int c_out = get_global_id(1);
    const int i = get_global_id(2);

    const int L_out = get_global_size(2); 

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K_size; ++k) {
            int l_in = (i * stride) + (k * dilation) - padding; 
            if (l_in >= 0 && l_in < L_in) {
                int idx_x = n * (C_in * L_in) + c_in * L_in + l_in;
                int idx_w = c_out * (C_in * K_size) + c_in * K_size + k;
                acc += x[idx_x] * weight[idx_w];
            }
        }
    }
    int idx_out = n * (get_global_size(1) * L_out) + c_out * L_out + i;
    output[idx_out] = acc;
}

__kernel void conv1d_backward_weight(
    const int stride, const int padding, const int dilation,
    const int N, const int L_in, const int L_out,
    __global const float *x,
    __global const float *grad_output,
    __global float *grad_weight)
{
    int c_out = get_global_id(0);
    int c_in  = get_global_id(1);
    int k     = get_global_id(2);

    int C_in   = get_global_size(1);
    int K_size = get_global_size(2);

    float sum = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int l_out = 0; l_out < L_out; ++l_out) {
            
            int l_in = l_out * stride + k * dilation - padding;
            
            if (l_in >= 0 && l_in < L_in) {
                int idx_out = n * (get_global_size(0) * L_out) + c_out * L_out + l_out;
                int idx_x   = n * (C_in * L_in) + c_in * L_in + l_in;
                
                sum += grad_output[idx_out] * x[idx_x];
            }
        }
    }

    int idx_gw = c_out * (C_in * K_size) + c_in * K_size + k;
    grad_weight[idx_gw] = sum;
}

__kernel void conv1d_backward_input(
    const int stride, const int padding, const int dilation,
    const int C_out, const int L_out, const int K_size,
    __global const float *grad_output,
    __global const float *weight,
    __global float *grad_input)
{
    int n    = get_global_id(0);
    int c_in = get_global_id(1);
    int l_in = get_global_id(2);

    int C_in = get_global_size(1);
    int L_in = get_global_size(2);

    float sum = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int k = 0; k < K_size; ++k) {
            int tmp = l_in - k * dilation + padding;
            if (tmp >= 0 && tmp % stride == 0) {
                int l_out = tmp / stride;
                if (l_out >= 0 && l_out < L_out) {
                    int idx_out = n * (C_out * L_out) + c_out * L_out + l_out;
                    int idx_w   = c_out * (C_in * K_size) + c_in * K_size + k;
                    sum += grad_output[idx_out] * weight[idx_w];
                }
            }
        }
    }
    int idx_gin = n * (C_in * L_in) + c_in * L_in + l_in;
    grad_input[idx_gin] = sum;
}
