from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader




class SIDD_Medium(Dataset):
    """Root
        - Image 1
          - Patch 000
            - GT_010.png
            - NOISY_010.png
            - GT_011.png
            - NOISY_011.png
        ....
    """
    def __init__(self, root_dir = r"./data/dir", 
                split_dir = None,
                transform=None):
        #root-dir: Root Directory of Dataset
        #transform(callable, optional): transform
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.transform = transforms.Compose([# resize
        transforms.ToTensor()         # totensor 
        ])
        self.split_dir = split_dir
        if split_dir is not None:
            with open(split_dir, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):  # 2줄씩 읽어서 pair로 만듦
                    gt_path = lines[i].strip()
                    noisy_path = lines[i + 1].strip()
                    self.image_pairs.append((gt_path, noisy_path))
        else:
            for image_folder in os.listdir(root_dir):
                image_folder_path = os.path.join(root_dir, image_folder)
                if not os.path.isdir(image_folder_path):
                    continue

                for patch_folder in os.listdir(image_folder_path):
                    patch_folder_path = os.path.join(image_folder_path, patch_folder)
                    if not os.path.isdir(patch_folder_path):
                        continue

                    gt_path_1 = os.path.join(patch_folder_path, "GT_SRGB_010.png")
                    noisy_path_1 = os.path.join(patch_folder_path, "NOISY_SRGB_010.png")
                
                    gt_path_2 = os.path.join(patch_folder_path, "GT_SRGB_011.png")
                    noisy_path_2 = os.path.join(patch_folder_path, "NOISY_SRGB_011.png")
                    # GT 및 노이즈 이미지가 모두 있는 경우만
                    if os.path.exists(gt_path_1) and os.path.exists(noisy_path_1):
                        self.image_pairs.append((gt_path_1, noisy_path_1))
                    
                    if os.path.exists(gt_path_2) and os.path.exists(noisy_path_2):
                        self.image_pairs.append((gt_path_2, noisy_path_2))                    
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]
            
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
            
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img
    

class RENOIR(Dataset):
    def __init__(self, root_dir = r"./data/dir", transform=None):
        self.root_dir = root_dir
        self.image_pairs = []        
        self.transform = transforms.Compose([                                          
        transforms.ToTensor()         # totensor 
        ])

        # 모든 Patch 폴더 경로 수집
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "Reference.bmp")
                noisy_path = os.path.join(patch_folder_path, "Noisy.bmp")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]    
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transform:
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, gt_img  # 입력: noisy, 타깃: reference


class PolyUDataset(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Denoising\Whole_Dataset_256\PolyU_Patched", transform=None):
        #root-dir: Root Directory of Dataset
        #transform(callable, optional): transform
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),  # resize
        transforms.ToTensor()         # totensor 
        ])
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "gt.jpg")
                noisy_path = os.path.join(patch_folder_path, "noisy.jpg")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]
            
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
            
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img

class RENOIR(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\RENOIR_Patched_Dataset", transform=None):
        self.root_dir = root_dir
        self.image_pairs = []        
        self.transform = transforms.Compose([                                          
        transforms.ToTensor()         # totensor 
        ])

        # 모든 Patch 폴더 경로 수집
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "Reference.bmp")
                noisy_path = os.path.join(patch_folder_path, "Noisy.bmp")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]    
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transform:
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, gt_img  # 입력: noisy, 타깃: reference


class DnD(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\PolyU_Patched", transform=None):
        #root-dir: Root Directory of Dataset
        #transform(callable, optional): transform
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.transform = transforms.Compose([  # resize
        transforms.ToTensor()         # totensor 
        ])
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "groundtruth.jpg")
                noisy_path = os.path.join(patch_folder_path, "noisy.jpg")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]
            
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
            
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img
    


class Custom(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Custom_Patched", transform=None):
        #root-dir: Root Directory of Dataset
        #transform(callable, optional): transform
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.transform = transforms.Compose([  # resize
        transforms.ToTensor()         # totensor 
        ])
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "gt.bmp")
                noisy_path = os.path.join(patch_folder_path, "input.bmp")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]
            
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
            
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, transform=None, sidd_split_dir=None):
        """
        통합 Denoising 데이터셋
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        
        # 기본 transform 설정
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # 각 데이터셋별로 이미지 쌍 수집
        self._load_sidd_dataset(sidd_split_dir)
        self._load_renoir_dataset()
        self._load_polyu_dataset()
        self._load_custom_dataset()
        
        print(f"총 {len(self.image_pairs)}개의 이미지 쌍을 로드했습니다.")
    
    def _load_sidd_dataset(self, split_dir=None):
        """SIDD 데이터셋 로드"""
        sidd_dir = os.path.join(self.root_dir, "SIDD_Patched")
        if not os.path.exists(sidd_dir):
            print(f"SIDD 폴더를 찾을 수 없습니다: {sidd_dir}")
            return
        
        sidd_count = 0
        
        if split_dir is not None:
            # split 파일이 있는 경우
            with open(split_dir, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):
                    gt_path = lines[i].strip()
                    noisy_path = lines[i + 1].strip()
                    if os.path.exists(gt_path) and os.path.exists(noisy_path):
                        self.image_pairs.append((gt_path, noisy_path))
                        sidd_count += 1
        else:
            # 전체 SIDD 데이터셋 로드
            for image_folder in os.listdir(sidd_dir):
                image_folder_path = os.path.join(sidd_dir, image_folder)
                if not os.path.isdir(image_folder_path):
                    continue
                
                for patch_folder in os.listdir(image_folder_path):
                    patch_folder_path = os.path.join(image_folder_path, patch_folder)
                    if not os.path.isdir(patch_folder_path):
                        continue
                    
                    # 010, 011 버전 모두 확인
                    for version in ["010", "011"]:
                        gt_path = os.path.join(patch_folder_path, f"GT_SRGB_{version}.png")
                        noisy_path = os.path.join(patch_folder_path, f"NOISY_SRGB_{version}.png")
                        
                        if os.path.exists(gt_path) and os.path.exists(noisy_path):
                            self.image_pairs.append((gt_path, noisy_path))
                            sidd_count += 1
        
        #print(f"SIDD: {sidd_count}개 이미지 쌍 로드")
    
    def _load_renoir_dataset(self):
        """RENOIR 데이터셋 로드"""
        renoir_dir = os.path.join(self.root_dir, "RENOIR_Patched")
        if not os.path.exists(renoir_dir):
            print(f"RENOIR 폴더를 찾을 수 없습니다: {renoir_dir}")
            return
        
        renoir_count = 0
        
        for image_folder in os.listdir(renoir_dir):
            image_folder_path = os.path.join(renoir_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue
            
            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue
                
                gt_path = os.path.join(patch_folder_path, "gt.bmp")
                noisy_path = os.path.join(patch_folder_path, "noisy.bmp")
                
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    renoir_count += 1
        
        #print(f"RENOIR: {renoir_count}개 이미지 쌍 로드")
    
    def _load_polyu_dataset(self):
        """PolyU 데이터셋 로드"""
        polyu_dir = os.path.join(self.root_dir, "PolyU_Patched")
        if not os.path.exists(polyu_dir):
            print(f"PolyU 폴더를 찾을 수 없습니다: {polyu_dir}")
            return
        
        polyu_count = 0
        
        for image_folder in os.listdir(polyu_dir):
            image_folder_path = os.path.join(polyu_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue
            
            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue
                
                gt_path = os.path.join(patch_folder_path, "gt.jpg")
                noisy_path = os.path.join(patch_folder_path, "noisy.jpg")
                
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    polyu_count += 1
        
        #print(f"PolyU: {polyu_count}개 이미지 쌍 로드")
    
    def _load_custom_dataset(self):
        """Custom 데이터셋 로드"""
        custom_dir = os.path.join(self.root_dir, "Custom_Patched")
        if not os.path.exists(custom_dir):
            print(f"Custom 폴더를 찾을 수 없습니다: {custom_dir}")
            return
        
        custom_count = 0
        
        for image_folder in os.listdir(custom_dir):
            image_folder_path = os.path.join(custom_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue
            
            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue
                
                gt_path = os.path.join(patch_folder_path, "gt.bmp")
                noisy_path = os.path.join(patch_folder_path, "noisy.bmp")
                
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    custom_count += 1
        
        #print(f"Custom: {custom_count}개 이미지 쌍 로드")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        gt_path, noisy_path = self.image_pairs[idx]
        
        # 이미지 로드 및 RGB 변환
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
        # Transform 적용
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img

    def get_dataset_info(self):
        info = {
            'total_pairs': len(self.image_pairs),
            'sidd_count': len([p for p in self.image_pairs if 'SIDD_Patched' in p[0]]),
            'renoir_count': len([p for p in self.image_pairs if 'RENOIR_Patched' in p[0]]),
            'polyu_count': len([p for p in self.image_pairs if 'PolyU_Patched' in p[0]]),
            'custom_count': len([p for p in self.image_pairs if 'Custom_Patched' in p[0]])
        }
        return info
    
if __name__ == "__main__":
    # 데이터셋 생성
    dataset = DenoisingDataset(
        root_dir=r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Denoising\Whole_Dataset_512",
    )
    
    # 데이터셋 정보 출력
    info = dataset.get_dataset_info()
    print("\n=== 데이터셋 정보 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 첫 번째 샘플 확인
    if len(dataset) > 0:
        noisy, gt = dataset[0]
        print(f"\n첫 번째 샘플 - Noisy: {noisy.shape}, GT: {gt.shape}")