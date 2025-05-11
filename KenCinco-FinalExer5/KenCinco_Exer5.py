import cv2
import numpy as np
import math

def create_vignette_mask(width, height, sigma=0.4, pulsating=False):
    center_x, center_y = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    dist_squared = ((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist_squared = (max(center_x, center_y) ** 2)
    normalized_dist_squared = dist_squared / max_dist_squared
    
    if not pulsating:
        mask = np.exp(-normalized_dist_squared / (2 * (sigma ** 2)))
    else:
        mask = normalized_dist_squared
        
    mask = np.clip(mask, 0.0, 1.0)
    
    return mask

def apply_vignette(frame, mask):
    frame_float = frame.astype(np.float32) / 255.0
    frame_vignetted = frame_float * mask
    frame_vignetted = np.clip(frame_vignetted * 255.0, 0, 255).astype(np.uint8)
    return frame_vignetted

def process_video_vignette(input_path, output_path, sigma=0.4, pulsating=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    is_color = len(first_frame.shape) == 3 and first_frame.shape[2] == 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
    
    base_mask = create_vignette_mask(width, height, sigma, pulsating)
    
    if is_color and len(base_mask.shape) == 2:
        base_mask = np.expand_dims(base_mask, axis=2)
        base_mask = np.repeat(base_mask, 3, axis=2)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if pulsating:
            progress = frame_idx / total_frames
            cycle_position = math.sin(2 * math.pi * progress * 2) * 0.2 + 0.8
            current_sigma = sigma * cycle_position
            current_mask = np.exp(-base_mask / (2 * (current_sigma ** 2)))
            current_mask = np.clip(current_mask, 0.0, 1.0)
            
            if is_color and len(current_mask.shape) == 2:
                current_mask = np.expand_dims(current_mask, axis=2)
                current_mask = np.repeat(current_mask, 3, axis=2)
                
            frame_with_vignette = apply_vignette(frame, current_mask)
        else:
            frame_with_vignette = apply_vignette(frame, base_mask)
        
        out.write(frame_with_vignette)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processing: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    print(f"Finished processing vignette effect. Output saved to: {output_path}")

input_path = 'my_test_video.mp4'

process_video_vignette(
    input_path, 
    'KenCinco-FinalExer5/transformed_video_exer5_vignette_static.mp4',
    sigma=0.4,
    pulsating=False
)

process_video_vignette(
    input_path, 
    'KenCinco-FinalExer5/transformed_video_exer5_vignette_pulsating.mp4',
    sigma=0.4,
    pulsating=True
)
