import torch
import torchvision.transforms as transforms
from PIL import Image
from arch.model import HybridSplitNet
import numpy as np
import torch.nn.functional as F
import cv2
import argparse




def preprocess_image(image_cv):

    # BGR to RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)    
    transform = transforms.ToTensor()
    image_tensor = transform(image_rgb).unsqueeze(0)
    
    return image_tensor


def postprocess_image(tensor):
    """
    Convert PyTorch tensor (RGB, CHW) to OpenCV image (BGR, HWC)
    """
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Clamp to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    image_np = tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    
    # Scale to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    # RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_bgr

def main():
    parser = argparse.ArgumentParser(description='Image Denoising with HybridSplitNet')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--noisy_image', type=str, required=True,
                        help='Path to the noisy input image')
    parser.add_argument('--gt_image', type=str, default=None,
                        help='Path to the ground truth image')
    parser.add_argument('--output_path', type=str, default='output.png',
                        help='Path to save the denoised output image')
    parser.add_argument('--ffn_expansion_factor', type=int, default=2,
                        help='FFN expansion factor for the model')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = HybridSplitNet(ffn_expansion_factor=args.ffn_expansion_factor)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    
    # Load images
    noisy_image = cv2.imread(args.noisy_image)
    input_tensor = preprocess_image(noisy_image).to(device)
    
    #Run Model
    print("Running denoising...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = postprocess_image(output_tensor)
    
    # Calculate PSNR

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    if args.gt_image is not None:
        gt_image = cv2.imread(args.gt_image)
        gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)    
        psnr = cv2.PSNR(gt_image_rgb,output_image_rgb)
        print(f"PSNR: {psnr:.2f} dB")
    
    # Save output image
    cv2.imwrite(args.output_path, output_image)
    print(f"Output image saved to {args.output_path}")
    
    # Display output image
    cv2.imshow("Denoised Output", output_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    
    
if __name__ == "__main__":
    main()