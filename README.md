**Intel GPU Support Now Available in PyTorch 2.5**

**by PyTorch Team at Intel**

Support for Intel GPUs is now available in PyTorch® 2.5, providing improved functionality and performance for Intel GPUs, including Intel® Arc™ discrete graphics, Intel® Core™ Ultra processors with built-in Intel® Arc™ graphics, and Intel® Data Center GPU Max Series. This integration brings Intel GPUs and the SYCL\* software stack into the official PyTorch stack, ensuring a consistent user experience and enabling more extensive AI application scenarios, particularly in the AI PC domain.

Developers and customers building for and using Intel GPUs will have a better user experience by directly obtaining continuous software support from native PyTorch, unified software distribution, and consistent product release time.

### Overview of Intel GPU Support

Intel GPU support in PyTorch provides eager mode and graph mode support in the PyTorch built-in front end. Eager mode now has an implementation of commonly used Aten operators with the SYCL programming language. Graph mode (torch.compile) now has an enabled Intel GPU back end to optimize for Intel GPUs and integrate Triton.

Essential components of Intel GPU support were added to PyTorch, including runtime, Aten operators, oneDNN, TorchInductor, Triton, and Intel GPU tool chains integration. Meanwhile, quantization and distributed features are being actively developed in preparation for the PyTorch 2.6 release.

### Features

PyTorch 2.5 features with an Intel GPU include:

- Inference and training workflows.
- Enhanced torch.compile and eager mode functionalities with improved performance.
- Data types such as FP32, BF16, FP16, and automatic mixed precision (AMP).
- Runs on Intel® Client GPUs and Intel® Data Center GPU Max Series.
- Supports Linux (Ubuntu, SUSE Linux, and Red Hat Linux) and Windows 10/11.

### Get Started

To use Intel GPU in PyTorch 2.5, follow these steps:

#### **1. Upgrade to PyTorch 2.5 (Nightly Build)**

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly
```

#### **2. Check if Intel GPU is Available**

```python
import torch

if torch.xpu.is_available():  # New Intel GPU check
    device = torch.device("xpu")
    print("Intel GPU is available!")
else:
    device = torch.device("cpu")
    print("No Intel GPU found, using CPU.")
```

#### **3. Run a Simple Tensor Test on Intel GPU**

```python
tensor = torch.tensor([1.0, 2.0]).to("xpu")  # Use "xpu" instead of "cuda"
print(tensor.device)
```

### **If Intel GPU is Still Not Detected**

- Check your driver version and update it if needed.
- Ensure you have **oneAPI** installed, as PyTorch 2.5 uses **SYCL** for Intel GPUs.
- Try installing **Intel’s extension for PyTorch**:
  ```bash
  pip install intel-extension-for-pytorch
  ```
  Then, check for GPU support:
  ```python
  import torch
  import intel_extension_for_pytorch as ipex

  print(torch.xpu.is_available())  # Should return True if the Intel GPU is detected
  ```

### **Performance**

Intel GPU on PyTorch was optimized to achieve strong performance on three Dynamo Hugging Face, TIMM, and TorchBench benchmarks for eager and compile modes.

Performance tests using Intel® Data Center GPU Max Series 1100 single card showed significant speedup ratios for FP16/BF16 over FP32 in eager mode, and Torch.compile mode demonstrated improvements over eager mode.

### **Summary**

Intel GPU on PyTorch 2.5 brings Intel® Client GPUs (Intel® Core™ Ultra processors with built-in Intel® Arc™ graphics and Intel® Arc™ Graphics for dGPU parts) and Intel® Data Center GPU Max Series into the PyTorch ecosystem for AI workload acceleration. Client GPUs are now supported for AI PC use scenarios on Windows and Linux environments.

The community is encouraged to evaluate and provide feedback on these enhancements to Intel GPU support in PyTorch.

### **Resources**

- PyTorch Docs: Getting Started on Intel GPU
- Intel® Tiber™ AI Cloud

