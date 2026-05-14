import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

def generate_kernel_registry(kernel_dir="kernels", output_header="kernel_registry.hpp"):
    cl_files = glob.glob(os.path.join(kernel_dir, "*.cl"))
    if not cl_files:
        print(f"[Warning] No *.cl file in '{kernel_dir}'")
    header_content = """#pragma once
#include <string>
#include <unordered_map>

namespace ht {
struct KernelRegistry {
    static std::unordered_map<std::string, std::string> get_sources() {
        return {
"""
    for cl_file in cl_files:
        filename = os.path.basename(cl_file)
        kernel_name = os.path.splitext(filename)[0] 
        with open(cl_file, 'r', encoding='utf-8') as f:
            source = f.read()
        header_content += f'            {{"{kernel_name}", R"CLC(\n{source}\n)CLC"}},\n'
    header_content += """        };
    }
};
} // namespace ht
"""
    with open(output_header, "w", encoding='utf-8') as f:
        f.write(header_content)

if not os.path.exists("kernels"):
    os.makedirs("kernels")
generate_kernel_registry()

setup(
    name='clAtenOps',
    ext_modules=[
        CppExtension(
            name='clAtenOps',
            sources=['main.cpp'],
            libraries=['OpenCL'],
            extra_compile_args=['-O3', '-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
