import os
import sys
import requests
import zipfile
from tqdm import tqdm
from urllib.parse import urlparse

# 可用数据集列表
AVAILABLE_DATASETS = [
    "ae_photos", "apple2orange", "summer2winter_yosemite", "horse2zebra",
    "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo",
    "maps", "cityscapes", "facades", "iphone2dslr_flower",
    "mini", "mini_pix2pix", "mini_colorization"
]


def show_available_datasets():
    """显示可用数据集列表"""
    print("可用数据集有: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
          "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, "
          "iphone2dslr_flower, ae_photos")


def download_file(url, filename):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 下载文件并显示进度条
        with open(filename, 'wb') as file, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        return True
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return False


def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python download_cyclegan_dataset.py [数据集名称]")
        show_available_datasets()
        sys.exit(1)

    dataset = sys.argv[1]

    # 验证数据集名称
    if dataset not in AVAILABLE_DATASETS:
        print(f"错误: 数据集 '{dataset}' 不可用")
        show_available_datasets()
        sys.exit(1)

    # 处理Cityscapes特殊情况
    if dataset == "cityscapes":
        print("由于许可证问题，我们无法从我们的存储库提供Cityscapes数据集。")
        print("请从 https://cityscapes-dataset.com 下载Cityscapes数据集，并使用脚本 ./datasets/prepare_cityscapes_dataset.py。")
        print(
            "你需要下载 gtFine_trainvaltest.zip 和 leftImg8bit_trainvaltest.zip。有关更多说明，请阅读 ./datasets/prepare_cityscapes_dataset.py")
        sys.exit(1)

    print(f"已指定 [{dataset}]")

    # 构建URL和文件路径
    url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset}.zip"
    zip_file = f"./datasets/{dataset}.zip"
    target_dir = f"./datasets/{dataset}/"

    # 下载文件
    print(f"正在从 {url} 下载...")
    if not download_file(url, zip_file):
        sys.exit(1)

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 解压文件
    print(f"正在解压 {zip_file} 到 {target_dir}...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("./datasets/")
        print("解压完成")
    except Exception as e:
        print(f"解压失败: {e}")
        sys.exit(1)

    # 删除ZIP文件
    os.remove(zip_file)
    print(f"已删除临时文件 {zip_file}")
    print(f"数据集已成功下载并提取到 {target_dir}")


if __name__ == "__main__":
    main()