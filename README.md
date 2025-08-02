# 🌟 Intel GPU Support in TensorFlow

## 🚀 Introduction
Intel® Extension for TensorFlow* is a high-performance deep learning extension that enhances TensorFlow with optimizations for Intel hardware, including CPUs and GPUs. This guide provides detailed installation steps for setting up TensorFlow with Intel GPU support.

---

## 🔧 System Requirements

### 🖥️ Hardware Requirements
- **Intel GPU** (e.g., Intel Data Center GPU Max Series, Intel Arc Graphics, etc.)
- **Compatible Intel CPU** (optional for hybrid execution)

### 📦 Software Requirements

#### Required Packages
| Package | CPU | GPU | Installation |
|---------|-----|-----|-------------|
| **Intel GPU Driver** | No | ✅ Yes | Install Intel GPU Driver |
| **Intel® oneAPI Base Toolkit** | No | ✅ Yes | Install Intel® oneAPI Base Toolkit |
| **TensorFlow** | ✅ Yes | ✅ Yes | Install TensorFlow 2.15.1 |

---

## 🛠️ Installation Steps

### 1️⃣ Install Intel GPU Driver
Ensure that the Intel GPU driver is installed correctly.
```bash
sudo apt update
sudo apt install intel-opencl-icd
sudo apt install level-zero
```
🔗 **For the latest drivers, refer to the** [Intel GPU driver documentation](https://dgpu-docs.intel.com/).

### 2️⃣ Install Intel® oneAPI Base Toolkit
The oneAPI Base Toolkit provides essential libraries for GPU acceleration.
```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/18412/l_BaseKit_p_2024.1.0.49192_offline.sh
chmod +x l_BaseKit_p_2024.1.0.49192_offline.sh
sudo ./l_BaseKit_p_2024.1.0.49192_offline.sh
```
🛑 **Follow the prompts to complete installation.**

### 3️⃣ Install TensorFlow with Intel GPU Support
Ensure you have Python installed (recommended: Python 3.8+).

#### 📌 Install TensorFlow
```bash
pip install tensorflow==2.15.1
```

#### 📌 Install Intel Extension for TensorFlow
```bash
pip install --upgrade intel-extension-for-tensorflow[xpu]
```

---

## 🌍 Environment Setup and Verification

### 🔹 Set Environment Variables
```bash
export TF_ENABLE_ONEDNN_OPTS=1
export TF_USE_XLA=1
export ZE_ENABLE_TRACING_LAYER=1
```

### 🔍 Verify Installation

#### ✅ Option 1: Check Environment
```bash
export path_to_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`
python ${path_to_site_packages}/intel_extension_for_tensorflow/tools/python/env_check.py
```

#### ✅ Option 2: Check TensorFlow Version
```bash
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```
🔹 **If the output displays the installed version of Intel Extension for TensorFlow, the installation is successful!** 🎉

---

## 📚 Additional Resources
- 📖 [Intel® Extension for TensorFlow* Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-tensorflow.html)
- 📖 [Intel GPU Driver Documentation](https://dgpu-docs.intel.com/)
- 📖 [Intel AI Developer Program](https://software.intel.com/ai)

---

# 🎯 Intel GPU Support Now Available in PyTorch 2.5

## 🚀 Introduction
Support for Intel GPUs is now available in **PyTorch® 2.5**, providing improved functionality and performance for:
- **Intel® Arc™ discrete graphics**
- **Intel® Core™ Ultra processors with built-in Intel® Arc™ graphics**
- **Intel® Data Center GPU Max Series**

This integration brings **Intel GPUs and the SYCL* software stack** into the official PyTorch stack, ensuring a consistent user experience and enabling extensive AI applications. 

---

## 🔥 Overview of Intel GPU Support

Intel GPU support in PyTorch provides **eager mode** and **graph mode** support. Key features include:

✅ **Implementation of commonly used Aten operators with SYCL**
✅ **Graph mode (`torch.compile`) optimized for Intel GPUs**
✅ **Integration with Triton, oneDNN, TorchInductor, and Intel GPU toolchains**

### ✨ Features
- **Inference and training workflows**
- **Enhanced `torch.compile` and eager mode functionalities**
- **Support for FP32, BF16, FP16, and AMP (Automatic Mixed Precision)**
- **Runs on Intel® Client GPUs and Intel® Data Center GPU Max Series**
- **Supported OS:** Linux (Ubuntu, SUSE Linux, Red Hat) and Windows 10/11

---

## 🏁 Getting Started

### 1️⃣ Upgrade to PyTorch 2.5 (Nightly Build)
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly
```

### 2️⃣ Check if Intel GPU is Available
```python
import torch

if torch.xpu.is_available():  # New Intel GPU check
    device = torch.device("xpu")
    print("Intel GPU is available!")
else:
    device = torch.device("cpu")
    print("No Intel GPU found, using CPU.")
```
## Damo
![image](https://github.com/user-attachments/assets/e59ac0f2-491a-4dcf-8fb2-dbf836628ad0)
### 3️⃣ Run a Simple Tensor Test on Intel GPU
```python
tensor = torch.tensor([1.0, 2.0]).to("xpu")  # Use "xpu" instead of "cuda"
print(tensor.device)
```

---

## 🔧 Troubleshooting

### 🛑 If Intel GPU is Still Not Detected
- **Check your driver version and update it if needed.**
- **Ensure you have oneAPI installed (required for PyTorch 2.5 with SYCL).**
- **Try installing Intel’s extension for PyTorch:**
  ```bash
  pip install intel-extension-for-pytorch
  ```
  Then, check for GPU support:
  ```python
  import torch
  import intel_extension_for_pytorch as ipex

  print(torch.xpu.is_available())  # Should return True if the Intel GPU is detected
  ```

---

## ⚡ Performance
Intel GPU support in PyTorch 2.5 delivers significant performance improvements:
✅ **Optimized for Hugging Face, TIMM, and TorchBench benchmarks**
✅ **FP16/BF16 show major speedup ratios over FP32**
✅ **`torch.compile` mode enhances performance further**

---

## 🎯 Summary
Intel GPU support in PyTorch 2.5 brings **Intel® Client GPUs** and **Intel® Data Center GPUs** into the PyTorch ecosystem for **AI workload acceleration**. Developers are encouraged to test, evaluate, and provide feedback. 🚀

### 🔗 Install Official Library for Intel GPU Support
```bash
pip install intel-extension-for-pytorch
```

---

## 📚 Additional Resources
- 📖 [PyTorch Docs: Getting Started on Intel GPU](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/)
- 📖 [Intel® Tiber™ AI Cloud](https://www.intel.com/content/www/us/en/developer/tools/tiber/ai-cloud.html)
- 📖 [Intel Extension for PyTorch on PyPI](https://pypi.org/project/intel-extension-for-pytorch/)


---

## 💼 Contact Me for Paid Projects

Have a project in mind or need expert help? I'm available for **freelance work and paid collaborations**. Let's bring your ideas to life with clean code and creative solutions.

📩 **Email**: [shuvobbhh@gmail.com]  
💬 **Telegram / WhatsApp**: [+8801616397082]  
🌐 **Portfolio / Website**: [[Portfolio](https://mahdi-hasan-shuvo.github.io/Mahdi-hasan-shuvo/)]

> *"Quality work speaks louder than words. Let's build something remarkable together."*
