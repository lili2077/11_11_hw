# 第 08 课作业

## 作业 01：用 RTX 5060 运行 PyTorch 训练

在装有 RTX 5060 的 Windows 电脑上完成。下面的命令每次输入一行，按 Enter，等运行结束再输入下一行。

### 1. 打开 Conda 命令行

**怎么做**　按键盘上的 Windows 键打开「开始」菜单。装了 Anaconda 或 Miniconda 的同学搜索 **Anaconda Prompt**，装了 Miniforge 的同学搜索 **Miniforge Prompt**。点击搜索结果，打开 Conda 自带的命令行窗口。 [Conda 官方说明](https://github.com/conda/conda/blob/main/docs/source/user-guide/install/windows.rst)

```bat
conda --version
```

**成功标志**　显示 `conda` 和版本号。两个入口都找不到且还没装 Conda，就先按官方说明安装 Windows 版。后续命令都在这个窗口运行。

### 2. 检查显卡驱动

**怎么做**　先运行 `nvidia-smi`。如果已经符合下方的成功标志，直接进入第 3 步。否则安装 **NVIDIA Studio Driver 621.14**（支持 Windows 10 64 位和 Windows 11）。按电脑类型打开下载页，点击 **Download**，运行下载的安装程序，重启电脑后再运行 `nvidia-smi`。

- [台式机 RTX 5060：下载 621.14](https://www.nvidia.com/en-us/geforce/drivers/details/283966/)
- [笔记本 RTX 5060 Laptop GPU：下载 621.14](https://www.nvidia.com/en-us/geforce/drivers/details/283967/)

```bat
nvidia-smi
```

**成功标志**　表格里列出 RTX 5060，`Driver Version` 为 `580.88` 或更高，`CUDA Version` 为 `13.0` 或更高。安装了上面的驱动后，`Driver Version` 应显示 `621.14`。PyTorch 自带运行库，无需另外安装 CUDA Toolkit。 [PyTorch 驱动要求](https://pytorch.org/blog/pytorch-2-12-release-blog/) · [运行库说明](https://discuss.pytorch.org/t/cuda-initialization-cuda-driver-initialization-failed/221081/7)

### 3. 创建并激活作业环境

**怎么做**　运行以下命令。如果已经创建过 `cv_gpu`，直接运行第二行。

```bat
conda create -n cv_gpu python=3.12 pip -y
conda activate cv_gpu
```

**成功标志**　命令行左侧出现 `(cv_gpu)`。

### 4. 安装并检查 GPU 版 PyTorch

**怎么做**　在 `(cv_gpu)` 环境中运行以下三行。第一行安装 CUDA 13.0 版 PyTorch，第二行安装 Jupyter，第三行检查 PyTorch 能否使用 CUDA。 [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/#v2141)

```bat
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install notebook ipykernel
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

**成功标志**　最后一条命令依次输出 `2.14.1+cu130`、`13.0` 和 `True`。如果显示 `None`，先确认命令行左侧是 `(cv_gpu)`，再检查是否安装了 `cu130` 版。如果显示 `13.0` 但最后一行是 `False`，回到第 2 步检查驱动。

### 5. 打开 Notebook 并选择内核

**怎么做**　在文件资源管理器中打开你保存 `01_verify_pytorch_gpu.ipynb` 的文件夹，点击顶部地址栏并复制完整路径。把下面第二行的 `你的作业文件夹路径` 替换成复制的路径，保留双引号，然后逐行执行。

```bat
python -m ipykernel install --user --name cv_gpu --display-name "Python (cv_gpu)"
cd /d "你的作业文件夹路径"
python -m notebook
```

在浏览器的文件列表中点击 [01_verify_pytorch_gpu.ipynb](01_verify_pytorch_gpu.ipynb)，在 `Kernel` → `Change Kernel` 中选择 **Python (cv_gpu)**。如果文件列表里没有作业，确认第二行填写的是 `.ipynb` 文件所在的文件夹路径。

**成功标志**　浏览器打开了作业，右上角显示 **Python (cv_gpu)**。第一格运行后打印的 Python 路径应包含 `cv_gpu`。如果提示找不到 `torch`，先核对内核和路径。

### 6. 运行并提交

**怎么做**　从上到下选中三个代码单元格，每格按 `Shift+Enter` 运行。运行完成后保存并提交带输出的 `.ipynb` 文件。

**成功标志**　三格都运行成功，并看到以下输出。

1. 第一格显示 RTX 5060 和计算能力 `(12, 0)`。 [NVIDIA 计算能力表](https://developer.nvidia.com/cuda/gpus)
2. 第二格显示结果在 `cuda:0`，矩阵左上角的值是 `256`。
3. 第三格显示数据、模型参数和梯度都在 `cuda:0`，训练后损失明显下降，斜率接近 `3`，截距接近 `-2`。

## 作业 02：CIFAR-10 分类与 GPU、CPU 耗时对比

### 1. 安装依赖

**怎么做**　完成作业 01 后，在 Conda 命令行运行以下命令。沿用同一个 `cv_gpu` 环境。 [PyTorch 与 TorchVision 版本搭配](https://pytorch.org/get-started/previous-versions/#v2141)

```bat
conda activate cv_gpu
python -m pip install torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install matplotlib
python -c "import torchvision, matplotlib; print(torchvision.__version__, matplotlib.__version__)"
```

**成功标志**　最后一行命令打印 TorchVision 和 Matplotlib 的版本号。

### 2. 运行并提交

**怎么做**　打开 [02_cifar10_gpu_vs_cpu.ipynb](02_cifar10_gpu_vs_cpu.ipynb)，选择 **Python (cv_gpu)** 内核，从上到下运行。首次运行会下载 CIFAR-10 到 `data/`。GPU 和 CPU 各训练 6 轮，等两次都结束后再看图表。

**成功标志**　数据数量依次为训练 45,000 张、验证 5,000 张和测试 10,000 张。Notebook 会输出两次训练的每轮指标、按验证集选出的模型的测试成绩，以及损失曲线、准确率曲线和耗时柱状图。

填写 Notebook 末尾的问题，保存带输出的 `.ipynb` 后提交。无需提交 `data/`。
