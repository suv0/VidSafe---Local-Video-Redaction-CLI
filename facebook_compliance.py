"""
Facebook-compliant video redaction system.
A clean, simple solution for creating platform-compliant videos.

This system mimics Facebook's automated content moderation to detect
and strategically blur content that would trigger their AI review.

Usage:
    python facebook_compliance.py input_video.mp4 output_video.mp4
"""

import cv2
import numpy as np
import argparse

def facebook_compliant_redaction(input_path: str, output_path: str):
    """
    Simple Facebook-compliant redaction that focuses on key violation types.
    This mimics what Facebook's AI would flag and blur accordingly.
    """
    print(f"📘 Facebook-Compliant Video Processing")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frames_processed = 0
    
    try:
        print("\n🎬 Processing with Facebook compliance...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')
            
            # Facebook-style content analysis
            facebook_risk_score = analyze_facebook_risk(frame)
            
            # If Facebook would flag this content, apply strategic blur
            if facebook_risk_score > 0.2:  # Facebook's likely threshold
                blur_mask = create_facebook_compliant_mask(frame, facebook_risk_score)
                redacted_frame = apply_facebook_blur(frame, blur_mask, facebook_risk_score)
                frames_processed += 1
            else:
                redacted_frame = frame.copy()
            
            # Write frame
            out.write(redacted_frame)
    
    finally:
        cap.release()
        out.release()
        
        print(f"\n\n✅ Facebook-Compliant Processing Complete!")
        print(f"Output saved to: {output_path}")
        print(f"Frames processed: {frames_processed}/{frame_count} ({100*frames_processed/frame_count:.1f}%)")


def analyze_facebook_risk(frame: np.ndarray) -> float:
    """
    Analyze Facebook content violation risk using computer vision.
    This mimics the types of analysis Facebook's AI likely performs.
    """
    # Convert to different color spaces
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    risk_score = 0.0
    
    # 1. Blood/Red content detection (Facebook flags graphic content)
    blood_score = detect_blood_content(hsv)
    risk_score = max(risk_score, blood_score * 0.8)  # High weight for blood
    
    # 2. Skin detection (potential NSFW or violence involving people)
    skin_score = detect_skin_content(hsv)
    risk_score = max(risk_score, skin_score * 0.6)  # Medium-high weight
    
    # 3. Edge/weapon detection (weapons, sharp objects)
    edge_score = detect_sharp_objects(gray)
    risk_score = max(risk_score, edge_score * 0.5)  # Medium weight
    
    # 4. Dark/disturbing scene detection
    darkness_score = detect_disturbing_scenes(gray)
    risk_score = max(risk_score, darkness_score * 0.4)  # Medium weight
    
    # 5. Central content analysis (Facebook focuses on central content)
    central_score = analyze_central_content(frame_rgb)
    risk_score = max(risk_score, central_score * 0.7)  # High weight for central content
    
    return min(risk_score, 1.0)


def detect_blood_content(hsv: np.ndarray) -> float:
    """Detect blood-like red content that Facebook would flag."""
    # Define red ranges for blood detection
    red_ranges = [
        ([0, 50, 50], [15, 255, 255]),    # Red range 1
        ([165, 50, 50], [180, 255, 255])  # Red range 2
    ]
    
    total_red_area = 0
    h, w = hsv.shape[:2]
    total_area = h * w
    
    for lower, upper in red_ranges:
        red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Filter out small regions (noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        red_area = np.sum(red_mask > 0)
        total_red_area += red_area
    
    # Calculate red content percentage
    red_percentage = total_red_area / total_area
    
    # Facebook likely flags content with significant red areas
    if red_percentage > 0.05:  # 5% red content
        return min(red_percentage * 10, 1.0)  # Scale up for Facebook sensitivity
    
    return red_percentage * 2


def detect_skin_content(hsv: np.ndarray) -> float:
    """Detect skin-tone content that might indicate people."""
    # Skin tone range
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = hsv.shape[:2]
    skin_area = np.sum(skin_mask > 0)
    skin_percentage = skin_area / (h * w)
    
    # Facebook flags content with significant human presence in certain contexts
    if skin_percentage > 0.1:  # 10% skin content
        return min(skin_percentage * 5, 1.0)
    
    return skin_percentage


def detect_sharp_objects(gray: np.ndarray) -> float:
    """Detect sharp edges that might indicate weapons or dangerous objects."""
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find long, straight lines (potential weapons)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    edge_score = 0.0
    
    if lines is not None:
        long_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 50:  # Long lines might be weapons
                long_lines += 1
        
        if long_lines > 5:  # Multiple long lines
            edge_score = min(long_lines / 20, 0.8)
    
    # Also check overall edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    if edge_density > 0.1:  # High edge density
        edge_score = max(edge_score, edge_density * 3)
    
    return edge_score


def detect_disturbing_scenes(gray: np.ndarray) -> float:
    """Detect dark or potentially disturbing scenes."""
    # Calculate overall brightness
    mean_brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    darkness_score = 0.0
    
    # Very dark scenes might indicate disturbing content
    if mean_brightness < 50:  # Very dark
        darkness_score = (50 - mean_brightness) / 50 * 0.6
    
    # High contrast might indicate dramatic/violent scenes
    if contrast > 60:  # High contrast
        darkness_score = max(darkness_score, (contrast - 60) / 60 * 0.4)
    
    return darkness_score


def analyze_central_content(frame_rgb: np.ndarray) -> float:
    """Analyze central content where Facebook focuses attention."""
    h, w = frame_rgb.shape[:2]
    
    # Extract central region (Facebook's likely focus area)
    center_h_start = h // 4
    center_h_end = 3 * h // 4
    center_w_start = w // 4
    center_w_end = 3 * w // 4
    
    central_region = frame_rgb[center_h_start:center_h_end, center_w_start:center_w_end]
    
    # Analyze central region for variance (activity/interest)
    gray_central = cv2.cvtColor(central_region, cv2.COLOR_RGB2GRAY)
    variance = np.var(gray_central)
    
    # High variance in central region might indicate important content
    if variance > 1000:  # High activity in center
        return min(variance / 5000, 0.8)
    
    return variance / 10000


def create_facebook_compliant_mask(frame: np.ndarray, risk_score: float) -> np.ndarray:
    """
    Create blur mask based on Facebook compliance requirements.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to analysis formats
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Strategy based on risk level
    if risk_score > 0.7:
        # High risk: Aggressive central blurring
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Add skin regions
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.bitwise_or(mask, skin_mask)
        
    elif risk_score > 0.4:
        # Medium risk: Target specific regions
        # Blood/red regions
        red_ranges = [([0, 50, 50], [15, 255, 255]), ([165, 50, 50], [180, 255, 255])]
        for lower, upper in red_ranges:
            red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, red_mask)
        
        # Central region
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.25)
        cv2.circle(mask, (center_x, center_y), radius, 128, -1)
        
    else:
        # Low risk: Light strategic blurring
        # Find areas of high variance
        kernel = np.ones((9, 9), np.float32) / 81
        gray_float = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_variance = cv2.filter2D((gray_float - local_mean) ** 2, -1, kernel)
        
        # Threshold high variance areas
        variance_threshold = np.percentile(local_variance, 90)
        high_variance_mask = (local_variance > variance_threshold).astype(np.uint8) * 128
        mask = cv2.bitwise_or(mask, high_variance_mask)
    
    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur for smooth transitions
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask


def apply_facebook_blur(frame: np.ndarray, mask: np.ndarray, risk_score: float) -> np.ndarray:
    """Apply Facebook-compliant blurring."""
    # Blur intensity based on risk score
    if risk_score > 0.7:
        blur_strength = 25  # Strong blur for high risk
    elif risk_score > 0.4:
        blur_strength = 15  # Medium blur
    else:
        blur_strength = 9   # Light blur
    
    # Create blurred version
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    
    # Apply mask
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_3d = np.dstack([mask_normalized] * 3)
    
    # Blend original and blurred based on mask
    result = frame.astype(np.float32) * (1 - mask_3d) + blurred.astype(np.float32) * mask_3d
    
    return result.astype(np.uint8)


def main():
    """Main entry point for Facebook-compliant video processing."""
    parser = argparse.ArgumentParser(description="Facebook-Compliant Video Redaction")
    
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    
    args = parser.parse_args()
    
    # Process video with Facebook compliance
    facebook_compliant_redaction(args.input, args.output)


if __name__ == "__main__":
    main()
