import torch
#
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def main():
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU 数量:", torch.cuda.device_count())
        print("当前 GPU 名称:", torch.cuda.get_device_name(0))
        print("当前设备索引:", torch.cuda.current_device())
    else:
        print("CUDA 不可用，正在使用 CPU。")

if __name__ == "__main__":
    main()