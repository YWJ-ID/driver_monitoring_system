#!/usr/bin/env python3
"""
自动下载驾驶员监控数据集的脚本
支持 StateFarm 和 DMD 数据集
"""

import os
import requests
import zipfile
import subprocess
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_statefarm_dataset():
    """下载 StateFarm 数据集"""
    print("正在下载 StateFarm Distracted Driver Detection Dataset...")
    
    # 检查是否已安装 kaggle
    try:
        subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
        print("使用 Kaggle CLI 下载...")
        subprocess.run([
            'kaggle', 'competitions', 'download', 
            '-c', 'state-farm-distracted-driver-detection'
        ])
        print("StateFarm 数据集下载完成！")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("未找到 Kaggle CLI，请手动下载：")
        print("1. 访问: https://www.kaggle.com/c/state-farm-distracted-driver-detection")
        print("2. 登录 Kaggle 账号")
        print("3. 点击 'Download' 按钮下载数据集")

def download_dmd_dataset():
    """下载 DMD 数据集"""
    print("正在下载 DMD (Driver Monitoring Dataset)...")
    
    # 创建目录
    os.makedirs('datasets', exist_ok=True)
    os.chdir('datasets')
    
    # 下载数据集
    dmd_url = "https://github.com/utiasSTARS/driver-monitoring-dataset/releases/download/v1.0/dmd.zip"
    annotations_url = "https://github.com/utiasSTARS/driver-monitoring-dataset/releases/download/v1.0/annotations.zip"
    
    try:
        print("下载 DMD 数据集...")
        download_file(dmd_url, 'dmd.zip')
        
        print("下载标注文件...")
        download_file(annotations_url, 'annotations.zip')
        
        # 解压文件
        print("解压数据集...")
        with zipfile.ZipFile('dmd.zip', 'r') as zip_ref:
            zip_ref.extractall('dmd')
        
        with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
            zip_ref.extractall('dmd')
        
        # 清理压缩文件
        os.remove('dmd.zip')
        os.remove('annotations.zip')
        
        print("DMD 数据集下载和解压完成！")
        print("数据集位置: datasets/dmd/")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载：")
        print("1. 访问: https://github.com/utiasSTARS/driver-monitoring-dataset")
        print("2. 下载 dmd.zip 和 annotations.zip")
        print("3. 解压到 datasets/dmd/ 目录")

def main():
    """主函数"""
    print("🚗 驾驶员监控数据集下载工具")
    print("=" * 50)
    
    while True:
        print("\n请选择要下载的数据集：")
        print("1. StateFarm Distracted Driver Detection Dataset")
        print("2. DMD (Driver Monitoring Dataset)")
        print("3. 下载所有数据集")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            download_statefarm_dataset()
        elif choice == '2':
            download_dmd_dataset()
        elif choice == '3':
            download_statefarm_dataset()
            download_dmd_dataset()
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 