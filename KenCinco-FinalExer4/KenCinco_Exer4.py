import cv2
import numpy as np
import math

def process_night_vision(input_path, output_path, add_scan_lines=True, noise_level=10):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    frame_idx = 0
    scan_line_period = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        progress = frame_idx / total_frames
        alpha = 1.2 + 0.2 * math.sin(2 * math.pi * progress * 0.5)
        beta = 10
        adjusted_gray = cv2.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)
        night_vision = np.zeros_like(frame)
        night_vision[:, :, 1] = adjusted_gray
        noise = np.random.normal(0, noise_level, adjusted_gray.shape).astype(np.int16)
        night_vision[:, :, 1] = np.clip(night_vision[:, :, 1].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if add_scan_lines and frame_idx % scan_line_period == 0:
            for y in range(0, height, 20):
                if y < height:
                    night_vision[y:y+2, :, :] = night_vision[y:y+2, :, :] // 3

        Y, X = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist_from_center / max_dist
        vignette = 1 - 0.3 * norm_dist**2
        night_vision[:, :, 1] = np.clip(night_vision[:, :, 1] * vignette, 0, 255).astype(np.uint8)

        out.write(night_vision)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Processing: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    cap.release()
    out.release()
    print(f"Finished processing night vision effect. Output saved to: {output_path}")

input_path = 'my_test_video.mp4'

process_night_vision(
    input_path, 
    'KenCinco-FinalExer4/transformed_video_exer4_with_scanlines.mp4',
    add_scan_lines=True,
    noise_level=15
)

process_night_vision(
    input_path, 
    'KenCinco-FinalExer4/transformed_video_exer4_no_scanlines.mp4',
    add_scan_lines=False,
    noise_level=10
)
