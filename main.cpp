#include "kernel_registry.hpp"

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <torch/torch.h>
#include <torch/extension.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>

namespace py = pybind11;

struct OpenCLBackend {
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;

    std::unordered_map<std::string, cl::Program> program_map;

    OpenCLBackend() {
        context = cl::Context(CL_DEVICE_TYPE_GPU);
        device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue = cl::CommandQueue(context, device);

        const auto& k_map = ht::KernelRegistry::get_sources();
        for (const auto& [k_file, source] : k_map) {
            cl::Program program(context, source);
            // TODO: check build error
            program.build();
            program_map[k_file] = program;
        }
    }

    cl::Program getProgram(const std::string& fileName) const {
        return program_map.at(fileName);
    }

    static OpenCLBackend& getInstance() {
        static OpenCLBackend instance;
        return instance;
    }
};

class Conv1dFunction : public torch::autograd::Function<Conv1dFunction> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor x, torch::Tensor weight,
        int64_t stride, int64_t padding, int64_t dilation)
    {
        ctx->save_for_backward({x, weight});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;

        x = x.to(torch::kFloat32).contiguous();
        weight = weight.contiguous();

        TORCH_CHECK(x.dim() >= 3, "x must have greater than 3 dimensions");

        // TODO: check x shape?

        int64_t C_in = x.size(x.dim() - 2);
        int64_t L_in = x.size(x.dim() - 1);
        int64_t N = x.numel() / C_in / L_in;

        int64_t C_out = weight.size(0);
        int64_t K_size = weight.size(2);

        int64_t L_out = (L_in + 2 * padding  - dilation * (K_size - 1) - 1) / stride + 1;

        std::vector<int64_t> output_sizes(x.sizes().begin(), x.sizes().end());
        output_sizes[(int)output_sizes.size() - 2] = C_out;
        output_sizes[(int)output_sizes.size() - 1] = L_out;
        auto output = torch::zeros(output_sizes, x.options());

        float *h_x = x.data_ptr<float>();
        float *h_w = weight.data_ptr<float>();
        float *h_out = output.data_ptr<float>();

        size_t x_bytes = x.numel() * sizeof(float);
        size_t w_bytes = weight.numel() * sizeof(float);
        size_t out_bytes = output.numel() * sizeof(float);

        // Call OpenCL kernel
        const auto& ocl = OpenCLBackend::getInstance();

        cl::Buffer d_x(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_bytes, h_x);
        cl::Buffer d_w(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_bytes, h_w);
        cl::Buffer d_out(ocl.context, CL_MEM_WRITE_ONLY, out_bytes);
        
        cl::Program program = ocl.getProgram("conv1d");
        cl::Kernel kernel(program, "conv1d");
        kernel.setArg(0, (int)stride);
        kernel.setArg(1, (int)dilation);
        kernel.setArg(2, (int)padding);
        kernel.setArg(3, (int)C_in);
        kernel.setArg(4, (int)L_in);
        kernel.setArg(5, (int)K_size);
        kernel.setArg(6, d_x);
        kernel.setArg(7, d_w);
        kernel.setArg(8, d_out);

        cl::NDRange global(N, C_out, L_out);
        ocl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        ocl.queue.enqueueReadBuffer(d_out, CL_TRUE, 0, out_bytes, h_out);
        ocl.queue.finish();

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto grad_output = grad_outputs[0].contiguous();
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto weight = saved[1];

        int64_t stride = ctx->saved_data["stride"].toInt();
        int64_t padding = ctx->saved_data["padding"].toInt();
        int64_t dilation = ctx->saved_data["dilation"].toInt();        

        auto grad_input = torch::zeros_like(x);
        auto grad_weight = torch::zeros_like(weight);
        int64_t C_in = x.size(x.dim() - 2);
        int64_t L_in = x.size(x.dim() - 1);
        int64_t N = x.numel() / C_in / L_in;
        int64_t C_out = weight.size(0);
        int64_t K_size = weight.size(2);
        int64_t L_out = grad_output.size(2);

        const auto& ocl = OpenCLBackend::getInstance();
        cl::Program program = ocl.getProgram("conv1d");
        
        cl::Buffer d_x(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, x.numel() * sizeof(float), x.data_ptr<float>());
        cl::Buffer d_w(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, weight.numel() * sizeof(float), weight.data_ptr<float>());
        cl::Buffer d_go(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, grad_output.numel() * sizeof(float), grad_output.data_ptr<float>());    
        cl::Buffer d_grad_in(ocl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, grad_input.numel() * sizeof(float), grad_input.data_ptr<float>());
        cl::Buffer d_grad_w(ocl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, grad_weight.numel() * sizeof(float), grad_weight.data_ptr<float>());

        cl::Kernel ker_gw(program, "conv1d_backward_weight");
        ker_gw.setArg(0, (int)stride); ker_gw.setArg(1, (int)padding); ker_gw.setArg(2, (int)dilation);
        ker_gw.setArg(3, (int)N); ker_gw.setArg(4, (int)L_in); ker_gw.setArg(5, (int)L_out);
        ker_gw.setArg(6, d_x); ker_gw.setArg(7, d_go); ker_gw.setArg(8, d_grad_w);
        cl::NDRange global_gw(C_out, C_in, K_size);
        ocl.queue.enqueueNDRangeKernel(ker_gw, cl::NullRange, global_gw, cl::NullRange);

        cl::Kernel ker_gi(program, "conv1d_backward_input");
        ker_gi.setArg(0, (int)stride); ker_gi.setArg(1, (int)padding); ker_gi.setArg(2, (int)dilation);
        ker_gi.setArg(3, (int)C_out); ker_gi.setArg(4, (int)L_out); ker_gi.setArg(5, (int)K_size);
        ker_gi.setArg(6, d_go); ker_gi.setArg(7, d_w); ker_gi.setArg(8, d_grad_in);
        cl::NDRange global_gi(N, C_in, L_in);
        ocl.queue.enqueueNDRangeKernel(ker_gi, cl::NullRange, global_gi, cl::NullRange);

        ocl.queue.finish();        
        return {grad_input, grad_weight, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

class Conv1dImpl : public torch::nn::Module {
  private:
    int64_t stride, padding, dilation;

  public:
    torch::Tensor weight;
    
    Conv1dImpl(int64_t in_ch, int64_t out_ch, int64_t k_size, int64_t stride, int64_t padding,
               int64_t dilation) : stride(stride), padding(padding), dilation(dilation)
    {
        // TODO: better weight initialization?
        weight = register_parameter("weight", torch::randn({out_ch, in_ch, k_size}));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return Conv1dFunction::apply(x, weight, stride, padding, dilation);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Conv1dImpl, std::shared_ptr<Conv1dImpl>>(m, "Conv1d")
            .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
                 py::arg("in_ch"), py::arg("out_ch"), py::arg("k_size"),
                 py::arg("stride") = 1, py::arg("padding") = 0, py::arg("dilation") = 1)
            .def("forward", &Conv1dImpl::forward)
	    .def("__call__", &Conv1dImpl::forward)
            .def_readwrite("weight", &Conv1dImpl::weight);
}
