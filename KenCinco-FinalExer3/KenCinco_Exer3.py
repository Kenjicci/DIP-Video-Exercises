import cv2
import numpy as np
import os

def create_test_video(output_path="test_video.mp4", duration=5, fps=30, size=(640, 480)):
    print(f"Creating a test video at {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        if not out.isOpened():
            out.release()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
            
        total_frames = duration * fps
        
        for i in range(total_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            center_x = int(size[0] * (0.5 + 0.3 * np.sin(i * 2 * np.pi / total_frames)))
            center_y = int(size[1] * (0.5 + 0.3 * np.cos(i * 2 * np.pi / total_frames)))
            cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)
            
            rect_x = int(size[0] * (0.5 + 0.3 * np.cos(i * 2 * np.pi / total_frames)))
            rect_y = int(size[1] * (0.5 + 0.3 * np.sin(i * 2 * np.pi / total_frames)))
            cv2.rectangle(frame, (rect_x-30, rect_y-30), (rect_x+30, rect_y+30), (0, 255, 0), -1)
            
            cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Test video created successfully at {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error creating test video: {str(e)}")
        return None

def apply_gradual_rotation(input_path, output_path_normal, output_path_scaled, final_angle=360):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} total frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_normal = cv2.VideoWriter(output_path_normal, fourcc, fps, (width, height))
    out_scaled = cv2.VideoWriter(output_path_scaled, fourcc, fps, (width, height))
    
    if not (out_normal.isOpened() and out_scaled.isOpened()):
        print("Warning: Could not initialize VideoWriter with mp4v codec. Trying XVID...")
        out_normal.release()
        out_scaled.release()
        
        output_path_normal = output_path_normal.replace('.mp4', '.avi')
        output_path_scaled = output_path_scaled.replace('.mp4', '.avi')
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_normal = cv2.VideoWriter(output_path_normal, fourcc, fps, (width, height))
        out_scaled = cv2.VideoWriter(output_path_scaled, fourcc, fps, (width, height))
        
        if not (out_normal.isOpened() and out_scaled.isOpened()):
            print("Error: Could not initialize VideoWriter with available codecs")
            return False

    center = (width // 2, height // 2)
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        progress = frame_idx / (total_frames - 1) if total_frames > 1 else 0
        current_angle = final_angle * progress
        
        rotation_matrix = cv2.getRotationMatrix2D(center, current_angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        out_normal.write(rotated_frame)
        
        angle_rad = np.abs(current_angle * np.pi / 180)
        
        scaling_factor = 1.0
        if current_angle % 90 != 0:
            cos_angle = np.abs(np.cos(angle_rad))
            sin_angle = np.abs(np.sin(angle_rad))
            new_width = width * cos_angle + height * sin_angle
            new_height = width * sin_angle + height * cos_angle
            scaling_factor = min(width / new_width, height / new_height)
        
        rotation_scale_matrix = cv2.getRotationMatrix2D(center, current_angle, scaling_factor)
        rotated_scaled_frame = cv2.warpAffine(frame, rotation_scale_matrix, (width, height))
        out_scaled.write(rotated_scaled_frame)
        
        if frame_idx % max(1, total_frames // 10) == 0:
            print(f"Processing: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%), "
                  f"Angle: {current_angle:.1f}°, Scale: {scaling_factor:.3f}")
    
    cap.release()
    out_normal.release()
    out_scaled.release()
    
    print(f"Normal rotation output saved to: {output_path_normal}")
    print(f"Scaled rotation output saved to: {output_path_scaled}")
    return True

if __name__ == "__main__":
    input_path = 'my_test_video.mp4'
    output_path_normal = 'KenCinco-FinalExer3/transformed_video_exer3_normal.mp4'
    output_path_scaled = 'KenCinco-FinalExer3/transformed_video_exer3_scaled.mp4'
    
    try:
        success = apply_gradual_rotation(input_path, output_path_normal, output_path_scaled, final_angle=360)
        if not success:
            print("Creating a test video...")
            test_video = create_test_video("my_test_video.mp4")
            if test_video:
                apply_gradual_rotation(test_video, output_path_normal, output_path_scaled, final_angle=360)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
