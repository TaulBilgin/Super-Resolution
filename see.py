import torch
import cv2
import os
import glob
import numpy as np
from utils import convert_image  # ### NEW CODE ### Import the utility function

# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
# ===================================================================

def rescale_frame(frame, real_w, real_h):
    return cv2.resize(frame, (real_w, real_h), interpolation=cv2.INTER_AREA)

def view_images_in_folder(folder_path):
    # Load the model once, before the loop starts
    srgan_generator = torch.load(srgan_checkpoint, weights_only=False)['generator'].to(device)
    srgan_generator.eval()
    model = srgan_generator

    """
    Loads, processes, and displays every image in a folder, one by one.
    """
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

    if not image_files:
        print(f"No .jpg images found in the folder: {folder_path}")
        return

    print("Starting image viewer...")
    print("Press ANY KEY to see the next image.")
    print("Press the 'ESC' key to quit.")

    # Loop through every image file in the list.
    for image_path in image_files:
        # Load the original high-resolution image
        hr_img = cv2.imread(image_path)
        if hr_img is None:
            print(f"Warning: Could not load image at {image_path}")
            continue

        # Create the low-resolution version for the model's input
        h, w, _ = hr_img.shape
        target_w, target_h = w // 4, h // 4
        lr_img = cv2.resize(hr_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # ### NEW CODE BLOCK - START: Model Processing ###

        # 1. PRE-PROCESS THE LR_IMG FOR THE MODEL
        # Convert NumPy BGR -> PIL RGB
        lr_img_pil = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        # Use utility function to convert PIL -> Normalized Tensor
        input_tensor = convert_image(lr_img_pil, source='pil', target='imagenet-norm')
        # Add batch dimension and move to GPU
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # 2. RUN INFERENCE
        with torch.no_grad():
            # Pass the prepared tensor through the model
            sr_tensor = model(input_tensor)

        # 3. POST-PROCESS THE OUTPUT TENSOR FOR VIEWING
        # Squeeze batch dimension, move to CPU
        sr_img_tensor = sr_tensor.squeeze(0).cpu()
        # Use utility function to convert Tensor [-1, 1] -> PIL Image
        sr_img_pil = convert_image(sr_img_tensor, source='[-1, 1]', target='pil')
        # Convert PIL Image -> NumPy BGR for OpenCV
        sr_img_bgr = cv2.cvtColor(np.array(sr_img_pil), cv2.COLOR_RGB2BGR)

        # ### NEW CODE BLOCK - END ###

        # Display the images in separate windows
        hr_img_window_title = 'Original High-Res'
        lr_img_window_title = 'Low-Res (Upscaled to View)'
        sr_img_window_title = 'Super-Resolved (SRGAN Output)' # ### NEW CODE ###

        cv2.imshow(hr_img_window_title, hr_img)
        cv2.imshow(lr_img_window_title, rescale_frame(lr_img, w, h))
        cv2.imshow(sr_img_window_title, sr_img_bgr) # ### NEW CODE ###

        # Wait for a key press.
        key = cv2.waitKey(0)

        # Destroy all windows before the next loop iteration
        cv2.destroyAllWindows()

        # If the 'Escape' key was pressed, exit the loop.
        if key == 27:
            print("Exit key pressed. Closing viewer.")
            break

    # Final cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Define the folder you want to view
    image_folder = "C:\\Users\\bilgi\\code2\\a-PyTorch-Tutorial-to-Super-Resolution-master\\val2014"

    # Call the function to start the viewer
    view_images_in_folder(image_folder)