# implement your oxford_pet.py
import os  # 用於處理檔案與目錄
import torch  # PyTorch 主函式庫
import shutil  # 用於解壓縮與檔案管理
import numpy as np  # Numpy，處理數據陣列
from PIL import Image  # 影像處理
from tqdm import tqdm  # 進度條顯示庫
from urllib.request import urlretrieve  # 下載網路上的數據
from torch.utils.data import Dataset, DataLoader  # PyTorch 的 Dataset 和 DataLoader


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset 處理類別
    - root: 資料集的根目錄
    - mode: "train", "valid", "test" 三種模式
    - transform: 可選的資料增強與預處理方法
    """
    def __init__(self, root, mode="train", transform=None):  # 初始化函式，設定數據集位置、模式、轉換函式
        assert mode in {"train", "valid", "test"}, "mode 必須為 'train', 'valid' 或 'test'"  # 確保 mode 合法
        self.root = root  # 設定數據集根目錄
        self.mode = mode  # 設定當前模式
        self.transform = transform  # 設定是否應用資料增強

        # 設定影像與 Mask 的路徑
        self.images_directory = os.path.join(self.root, "images")  # 圖像資料夾
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")  # 標註資料夾

        self.filenames = self._read_split()  # 讀取訓練/驗證/測試的檔案名稱

    def __len__(self):
        return len(self.filenames)  # 回傳數據集長度

    def __getitem__(self, idx):
        """讀取一個樣本"""
        filename = self.filenames[idx]  # 取得對應索引的檔名
        image_path = os.path.join(self.images_directory, filename + ".jpg")  # 影像路徑
        mask_path = os.path.join(self.masks_directory, filename + ".png")  # Mask 路徑

        image = np.array(Image.open(image_path).convert("RGB"))  # 讀取影像並轉換為 RGB
        trimap = np.array(Image.open(mask_path))  # 讀取 Trimap

        mask = self._preprocess_mask(trimap)  # 轉換 Trimap 為二元 Mask

        sample = dict(image=image, mask=mask, trimap=trimap)  # 組成字典
        if self.transform:
            sample = self.transform(**sample)  # 若有 transform，則應用

        return sample


    @staticmethod
    def _preprocess_mask(mask):
        """
        將三分類 trimap 轉換為二元 segmentation mask：
        - label 1 和 3 設為前景 (1)
        - label 2 設為背景 (0)
        """
        mask = mask.astype(np.float32)  # 轉換為浮點數格式
        mask[mask == 2] = 0  # 背景設為 0
        mask[(mask == 1) | (mask == 3)] = 1  # 物體 (貓/狗) 設為 1
        return mask

    def _read_split(self):
            """
            讀取資料集分割 (train / valid / test)
            - train: 80% 資料
            - valid: 10% 資料
            - test: 10% 資料 (官方提供的 test.txt)
            """
            # 使用 trainval.txt 中的資料
            split_filename = "trainval.txt"  # 包含所有訓練和驗證資料
            split_filepath = os.path.join(self.root, "annotations", split_filename)

            with open(split_filepath) as f:
                split_data = f.read().strip().split("\n")

            filenames = [x.split(" ")[0] for x in split_data]

            # 將 80% 訓練，10% 驗證，10% 測試
            num_samples = len(filenames)
            num_train = int(0.8 * num_samples)  # 80% 訓練集
            num_valid = int(0.1 * num_samples)  # 10% 驗證集

            # 隨機打亂資料
            np.random.shuffle(filenames)

            # 根據分割將資料分配到訓練、驗證和測試集
            if self.mode == "train":
                filenames = filenames[:num_train]  # 訓練集：前 80%
            elif self.mode == "valid":
                filenames = filenames[num_train:num_train + num_valid]  # 驗證集：80% 到 90%
            elif self.mode == "test":
                filenames = filenames[num_train + num_valid:]  # 測試集：90% 到 100%

            return filenames

    @staticmethod
    def download(root):
        """
        下載並解壓縮 Oxford-IIIT Pet Dataset
        """
        # 下載圖片數據集
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # 下載標註數據集
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

class SimpleOxfordPetDataset(OxfordPetDataset):
    """
    進一步處理 Oxford-IIIT Pet Dataset：
    - 影像與 Mask 皆調整為 256x256
    - 轉換為 PyTorch tensor 格式
    """
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
         # **取得圖像檔案名稱**
        image_filename = self.filenames[idx] + ".jpg"
        image_path = os.path.join(self.images_directory, image_filename)  # 取得完整路徑


        # 影像 & Mask resize 到 256x256
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # 轉換 HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)  # (H, W, C) -> (C, H, W)
        sample["mask"] = np.expand_dims(mask, 0)  # (H, W) -> (1, H, W)
        sample["trimap"] = np.expand_dims(trimap, 0)  # (H, W) -> (1, H, W)
        
        # **新增圖像路徑**
        sample["image_path"] = image_path  # ✅ 加入原始圖片的完整路徑

        return sample

class TqdmUpTo(tqdm):
    """
    Tqdm 進度條 (下載檔案時顯示)
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    """
    下載檔案
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n

def extract_archive(filepath):
    """
    解壓縮 .tar.gz 檔案
    """
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, batch_size=16, num_workers=4):
    """
    加載 Oxford-IIIT Pet Dataset，回傳 DataLoader
    """
    train_dataset = SimpleOxfordPetDataset(root=data_path, mode="train")
    valid_dataset = SimpleOxfordPetDataset(root=data_path, mode="valid")
    test_dataset = SimpleOxfordPetDataset(root=data_path, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    data_path = "dataset"

    # 如果資料集未下載，則下載
    if not os.path.exists(os.path.join(data_path, "images")):
        print("下載資料集...")
        OxfordPetDataset.download(data_path)

    # 測試 DataLoader 是否正常
    train_loader, valid_loader, test_loader = load_dataset(data_path)

    for batch in train_loader:
        images, masks = batch["image"], batch["mask"]
        print(f"影像形狀: {images.shape}, Mask 形狀: {masks.shape}")
        break

