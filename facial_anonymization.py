import cv2
import dlib
import numpy as np

def process_video_with_improved_mask():

    input_path = "test.mov"
    output_path = "output_blured.mp4"
    
    # Load dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Processing video with {total_frames} frames...")
    frame_count = 0
    
    # Variables for tracking
    last_valid_mask = None
    tracking_failure_count = 0
    max_tracking_failures = 30  # Number of frames to keep using the last valid mask
    
    # For smoother transitions
    smoothing_frames = 5
    previous_masks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:  # Show progress every 30 frames
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames*100):.1f}%)")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask for this frame
        height, width = frame.shape[:2]
        current_mask = np.zeros((height, width), np.uint8)
        
        # Detect faces with different parameters for better detection
        faces = detector(gray, 1)  # The second parameter increases detection sensitivity
        
        mask_updated = False
        
        if len(faces) > 0:
            tracking_failure_count = 0  # Reset failure counter
            
            for face in faces:
                # Get facial landmarks
                landmarks = predictor(gray, face)
                
                # 1. Create eyebrow region (landmarks 17-26)
                left_eyebrow = []
                right_eyebrow = []
                
                for i in range(17, 22):  # Left eyebrow
                    left_eyebrow.append((landmarks.part(i).x, landmarks.part(i).y))
                    
                for i in range(22, 27):  # Right eyebrow
                    right_eyebrow.append((landmarks.part(i).x, landmarks.part(i).y))
                
                # 2. Create eye region (landmarks 36-47)
                left_eye = []
                right_eye = []
                
                for i in range(36, 42):  # Left eye
                    left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
                    
                for i in range(42, 48):  # Right eye
                    right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
                
                # 3. Create forehead region
                # Get the highest points of eyebrows
                left_eyebrow_top = min([y for _, y in left_eyebrow])
                right_eyebrow_top = min([y for _, y in right_eyebrow])
                eyebrow_top = min(left_eyebrow_top, right_eyebrow_top)
                
                # Get leftmost and rightmost points of eyebrows
                leftmost = min([x for x, _ in left_eyebrow])
                rightmost = max([x for x, _ in right_eyebrow])
                
                # Calculate forehead height
                eye_top = min([y for _, y in left_eye + right_eye])
                eyebrow_eye_distance = eye_top - eyebrow_top
                forehead_height = int(eyebrow_eye_distance * 1.8)  # Increased for better coverage
                
                # Create forehead points
                forehead_top = max(0, eyebrow_top - forehead_height)
                forehead_points = [
                    (leftmost - 20, forehead_top),  # Left edge with more padding
                    (rightmost + 20, forehead_top)  # Right edge with more padding
                ]
                
                # 4. Combine regions into a single polygon
                upper_face_points = [forehead_points[0]]
                upper_face_points.append(forehead_points[1])
                upper_face_points.extend(right_eyebrow[::-1])
                upper_face_points.extend(right_eye[::-1])
                upper_face_points.extend(left_eye)
                upper_face_points.extend(left_eyebrow)
                
                # Fill the polygon on the mask
                cv2.fillPoly(current_mask, [np.array(upper_face_points)], 255)
                
                # Ensure eyebrows and eyes are fully covered with larger regions
                # Expand eye regions for better coverage
                def expand_region(points, expand_factor=1.3):
                    # Find center
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    
                    # Expand points from center
                    expanded = []
                    for x, y in points:
                        dx = x - center_x
                        dy = y - center_y
                        expanded.append((int(center_x + dx * expand_factor), 
                                        int(center_y + dy * expand_factor)))
                    return expanded
                
                # Expand and fill eye and eyebrow regions
                cv2.fillPoly(current_mask, [np.array(expand_region(left_eyebrow, 1.4))], 255)
                cv2.fillPoly(current_mask, [np.array(expand_region(right_eyebrow, 1.4))], 255)
                cv2.fillPoly(current_mask, [np.array(expand_region(left_eye, 1.5))], 255)
                cv2.fillPoly(current_mask, [np.array(expand_region(right_eye, 1.5))], 255)
                
                # Draw a rectangle for the forehead with more padding
                cv2.rectangle(current_mask, 
                             (leftmost - 20, forehead_top), 
                             (rightmost + 20, eyebrow_top), 
                             255, -1)
                
                # Dilate the mask to ensure complete coverage
                kernel = np.ones((7, 7), np.uint8)  # Larger kernel for more dilation
                current_mask = cv2.dilate(current_mask, kernel, iterations=3)  # More iterations
                
                mask_updated = True
        
        # If no face detected or landmarks failed, use the last valid mask
        if not mask_updated:
            tracking_failure_count += 1
            if last_valid_mask is not None and tracking_failure_count <= max_tracking_failures:
                current_mask = last_valid_mask.copy()
                print(f"No face detected in frame {frame_count}, using last valid mask")
            else:
                print(f"No face detected in frame {frame_count} and no valid mask available")
        else:
            last_valid_mask = current_mask.copy()
        
        # Add current mask to the list of previous masks for smoothing
        previous_masks.append(current_mask)
        if len(previous_masks) > smoothing_frames:
            previous_masks.pop(0)
        
        # Create a smoothed mask by averaging the last few masks
        smoothed_mask = np.zeros_like(current_mask, dtype=np.float32)
        for mask in previous_masks:
            smoothed_mask += mask.astype(np.float32)
        smoothed_mask = (smoothed_mask / len(previous_masks)).astype(np.uint8)
        
        # Threshold the smoothed mask to make it binary again
        _, final_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create a black frame
        black = np.zeros_like(frame)
        
        # Apply the black color to the masked region
        masked_black = cv2.bitwise_and(black, black, mask=final_mask)
        inverse_mask = cv2.bitwise_not(final_mask)
        background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        frame = cv2.add(background, masked_black)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processing complete. Output saved as: {output_path}")

if __name__ == "__main__":
    print("Starting improved black mask for forehead, eyebrows, and eyes...")
    print("Input file: test.mp4")
    print("Output file: output_blured.mp4")
    process_video_with_improved_mask()
    print("Done!")