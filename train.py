from arch.model import HybridSplitNet, CNNSplitNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary 
from torchvision import transforms
from data.dataset import SIDD_Medium, DenoisingDataset, PolyUDataset
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import param.lr_scheduler
import os
import glob
from skimage.metrics import structural_similarity as ssim
from param.losses import PSNRLoss, CombinedLoss, CharbonnierLoss
from torch.utils.data import random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HybridSplitNet()
model.to(device)

# Init Dataset
dataset_dir = r"D:\Dataset\Denoising\Whole_Dataset_256\SIDD_Patched"



dataset = SIDD_Medium(root_dir=dataset_dir, transform=None)
data_size = len(dataset)
print(f"Total dataset size: {data_size}")

train_size = int(0.8 * data_size)
val_size = int(0.1 * data_size)
test_size = data_size - train_size - val_size
batch_size = 4

import torch

memory_info = torch.cuda.memory_allocated()
print(memory_info)




train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("New Dataset prepared")

# Parameter
criterion = PSNRLoss()
criterion.cuda()

optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=2e-4, weight_decay=1e-4)

# ✅ Iteration 설정
total_iterations = 300000  # 200K iterations
iterations_per_epoch = len(train_loader)
estimated_epochs = total_iterations / iterations_per_epoch

print(f"\n=== Training Configuration ===")
print(f"Total iterations: {total_iterations}")
print(f"Iterations per epoch: {iterations_per_epoch}")
print(f"Estimated epochs: {estimated_epochs:.1f}")
print(f"Batch size: {batch_size}")

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-6)

# PSNR
def calculate_psnr(img1, img2):
    """img1, img2 range [0,1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# SSIM
def calculate_ssim(img1, img2):
    img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)

# Checkpoint
use_checkpoint = input("체크포인트를 불러올까요? (y/n): ").strip().lower() == 'y'

checkpoint_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Code\Denoising_Project-main\Model\Custom"
checkpoint_files = 'checkpoint_iter_best.pth'

current_iter = 0
start_epoch = 0

if use_checkpoint:
    if checkpoint_files and os.path.exists(checkpoint_files):
        checkpoint = torch.load(checkpoint_files, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # ✅ Iteration 복원
        if 'iteration' in checkpoint:
            current_iter = checkpoint['iteration']
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Checkpoint loaded from iteration {current_iter}, epoch {start_epoch}")
        else:
            print("Checkpoint loaded, but starting from iteration 0")
    else:
        print("No Checkpoint file found. Starting from scratch")
else:
    print("Starting from scratch")

# Early Stopping & Best tracking
best_psnr = 0
patience = 10  # epoch 단위
patience_counter = 0

# ✅ Training Loop
max_iter = total_iterations
validation_interval = iterations_per_epoch  # 매 epoch마다 validation

for epoch in range(start_epoch, 10000):  # 충분히 큰 수
    model.train()
    train_loss = 0.0
    epoch_start_iter = current_iter
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} - Iteration {current_iter}/{max_iter}")
    print(f"{'='*60}")
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr:.2e}")
    
    for noisy_images, gt_images in tqdm(train_loader, desc=f"Training", leave=False):
        if current_iter >= max_iter:
            print(f"\n✓ Reached maximum iterations: {max_iter}")
            break
        
        noisy_images, gt_images = noisy_images.to(device), gt_images.to(device)
        
        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, gt_images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        current_iter += 1
        
        # ✅ 주기적 loss 출력
        if current_iter % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = train_loss / (current_iter - epoch_start_iter)
            # PSNR 계산 (현재 배치)
            with torch.no_grad():
                outputs_clamp = torch.clamp(outputs, 0.0, 1.0)
                gt_clamp = torch.clamp(gt_images, 0.0, 1.0)
                batch_psnr = calculate_psnr(outputs_clamp, gt_clamp)
            
            print(f"Iter {current_iter}/{max_iter} - Loss: {avg_loss:.4f}, PSNR: {batch_psnr:.2f} dB, LR: {current_lr:.2e}")
    
    if current_iter >= max_iter:
        print("\nTraining completed!")
        break
    
    avg_loss = train_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Iterations completed: {current_iter}/{max_iter}")
    
    # ✅ Validation
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for noisy_images, gt_images in tqdm(val_loader, desc="Validating", leave=False):
            noisy_images, gt_images = noisy_images.to(device), gt_images.to(device)
            outputs = model(noisy_images)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            gt_images = torch.clamp(gt_images, 0.0, 1.0)

            for i in range(outputs.size(0)):
                out_img = outputs[i:i+1]
                gt_img = gt_images[i:i+1]

                psnr = calculate_psnr(out_img, gt_img)
                ssim_val = calculate_ssim(out_img, gt_img)

                total_psnr += psnr
                total_ssim += ssim_val

    avg_psnr = total_psnr / val_size
    avg_ssim = total_ssim / val_size
    
    print(f"  Validation PSNR: {avg_psnr:.2f} dB")
    print(f"  Validation SSIM: {avg_ssim:.4f}")
    
    # ✅ Best model 저장
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        patience_counter = 0
        
        best_checkpoint = {
            'epoch': epoch + 1,
            'iteration': current_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr': best_psnr,
            'ssim': avg_ssim,
            'loss': avg_loss
        }
        torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_model_iter.pth'))
        print(f"  ✓ New Best PSNR: {best_psnr:.2f} dB (saved)")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}")
    
    # Early Stopping
    if patience_counter >= patience:
        print(f"\n=== Early Stopping at Epoch {epoch+1} ===")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        break
    
    # ✅ Checkpoint 저장 (10K iteration마다)
    if current_iter % 10000 == 0 and current_iter > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{current_iter}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'iteration': current_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'psnr': avg_psnr
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved at iteration {current_iter}")
    
    # Temporary checkpoint (매 epoch)
    checkpoint = {
        'epoch': epoch + 1,
        'iteration': current_iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'psnr': avg_psnr
    }
    torch.save(checkpoint, "hybrid_model_tmp.pth")

print(f"\n{'='*60}")
print(f"=== Training Complete ===")
print(f"{'='*60}")
print(f"Total iterations: {current_iter}")
print(f"Best Validation PSNR: {best_psnr:.2f} dB")

# ✅ Final model 저장
torch.save(model.state_dict(), "hybrid_model_final.pth")
print("Final model saved at hybrid_model_final.pth")

# ✅ Test Evaluation
print("\n=== Evaluating on Test Dataset ===")
model.eval()
total_psnr = 0.0
total_ssim = 0.0

with torch.no_grad():
    for noisy_images, gt_images in tqdm(test_loader, desc="Testing"):
        noisy_images = noisy_images.to(device)
        gt_images = gt_images.to(device)

        outputs = model(noisy_images)
        outputs = torch.clamp(outputs, 0.0, 1.0)
        gt_images = torch.clamp(gt_images, 0.0, 1.0)

        for i in range(outputs.size(0)):
            out_img = outputs[i:i+1]
            gt_img = gt_images[i:i+1]

            psnr = calculate_psnr(out_img, gt_img)
            ssim_val = calculate_ssim(out_img, gt_img)

            total_psnr += psnr
            total_ssim += ssim_val

avg_psnr = total_psnr / test_size
avg_ssim = total_ssim / test_size

print(f"\n{'='*60}")
print(f"=== Test Results ===")
print(f"{'='*60}")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")