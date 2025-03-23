# Facial Anonymization Tool

A Python script for video processing that anonymizes a person's face by applying a black mask to the forehead, eyebrows, and eyes. The tool uses computer vision techniques to detect faces, identify facial landmarks, and create a precise mask that follows facial features throughout the video.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Customization](#customization)

## Overview

This tool processes video files to anonymize faces by:

- Detecting faces in each frame
- Identifying precise facial landmarks (eyes, eyebrows, etc.)
- Creating a black mask covering the forehead, eyebrows, and eyes
- Maintaining consistent masking even when face detection temporarily fails
- Producing a processed video with anonymized faces

## Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - OpenCV
  - dlib
  - NumPy

### Setup

1. Install the required packages:

```bash
pip install opencv-python dlib numpy
```

## How It Works

### Main Processing Steps

1. **Face Detection**: Identifies faces in each video frame
2. **Landmark Identification**: Locates 68 specific points on each face
3. **Region Extraction**: Isolates the forehead, eyebrows, and eyes
4. **Mask Creation**: Generates a precise mask for these regions
5. **Mask Application**: Applies a black color to the masked areas
6. **Tracking & Smoothing**: Maintains consistent masking between frames
Thought for 3 secondsYou're right - let me provide a more comprehensive README with all the details:

```markdown
# Facial Anonymization Tool

A Python script for video processing that anonymizes a person's face by applying a black mask to the forehead, eyebrows, and eyes. The tool uses computer vision techniques to detect faces, identify facial landmarks, and create a precise mask that follows facial features throughout the video.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Complete Code](#complete-code)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Customization](#customization)
- [Performance Considerations](#performance-considerations)

## Overview

This tool processes video files to anonymize faces by:

- Detecting faces in each frame
- Identifying precise facial landmarks (eyes, eyebrows, etc.)
- Creating a black mask covering the forehead, eyebrows, and eyes
- Maintaining consistent masking even when face detection temporarily fails
- Producing a processed video with anonymized faces

## Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - OpenCV
  - dlib
  - NumPy

### Setup

1. Install the required packages:

```bash
pip install opencv-python dlib numpy
```

2. Download the dlib facial landmark predictor:


```shellscript
# Download the shape predictor file
curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2

# Extract the file
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

3. Place the `shape_predictor_68_face_landmarks.dat` file in the same directory as the script.


## Usage

1. Place your input video file named `test.mp4` in the same directory as the script.
2. Run the script:


```shellscript
python facial_anonymization.py
```

3. The processed video will be saved as `output_blured.mp4` in the same directory.


## How It Works

### Main Processing Steps

1. **Face Detection**: Identifies faces in each video frame
2. **Landmark Identification**: Locates 68 specific points on each face
3. **Region Extraction**: Isolates the forehead, eyebrows, and eyes
4. **Mask Creation**: Generates a precise mask for these regions
5. **Mask Application**: Applies a black color to the masked areas
6. **Tracking & Smoothing**: Maintains consistent masking between frames


### Detailed Workflow

#### Initialization Phase

```python
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

- The script initializes dlib's face detector and loads a pre-trained facial landmark predictor model
- This model identifies 68 specific points on a human face (eyes, eyebrows, nose, mouth, jawline)


#### Video Processing Setup

```python
cap = cv2.VideoCapture(input_path)
# ... get video properties ...
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
```

- Opens the input video file and creates a VideoWriter to save the processed output
- Preserves the original video's resolution and frame rate


#### Tracking Variables

```python
last_valid_mask = None
tracking_failure_count = 0
max_tracking_failures = 30
smoothing_frames = 5
previous_masks = []
```

- These variables help maintain consistent masking when face detection fails
- `last_valid_mask`: Stores the most recent successful mask
- `tracking_failure_count`: Counts consecutive frames where face detection failed
- `previous_masks`: Stores recent masks for smoothing between frames


#### Main Processing Loop

For each frame in the video:

1. **Face Detection**

```python
faces = detector(gray, 1)
```

1. Converts the frame to grayscale and detects faces
2. The parameter `1` increases detection sensitivity



2. **Facial Landmark Identification**

```python
landmarks = predictor(gray, face)
```

1. For each detected face, identifies 68 facial landmarks
2. These landmarks are used to precisely locate facial features



3. **Extracting Key Facial Regions**

```python
# Extract eyebrow points (landmarks 17-26)
for i in range(17, 22):  # Left eyebrow
    left_eyebrow.append((landmarks.part(i).x, landmarks.part(i).y))
# ... similar for right eyebrow and eyes ...
```

1. Extracts specific landmark points for:

1. Left eyebrow (points 17-21)
2. Right eyebrow (points 22-26)
3. Left eye (points 36-41)
4. Right eye (points 42-47)






4. **Creating the Forehead Region**

```python
eyebrow_top = min(left_eyebrow_top, right_eyebrow_top)
forehead_height = int(eyebrow_eye_distance * 1.8)
forehead_top = max(0, eyebrow_top - forehead_height)
```

1. Calculates the forehead region based on eyebrow position
2. Uses the distance between eyes and eyebrows to estimate appropriate forehead height



5. **Region Expansion for Better Coverage**

```python
def expand_region(points, expand_factor=1.3):
    # Find center and expand points from center
    # ...
```

1. Expands eye and eyebrow regions by scaling them from their center
2. Ensures complete coverage of these features
3. Left/right eyes expanded by 50% (factor 1.5)
4. Eyebrows expanded by 40% (factor 1.4)



6. **Mask Creation and Enhancement**

```python
cv2.fillPoly(current_mask, [np.array(upper_face_points)], 255)
# ... additional region filling ...
kernel = np.ones((7, 7), np.uint8)
current_mask = cv2.dilate(current_mask, kernel, iterations=3)
```

1. Creates a binary mask where white (255) represents areas to be blacked out
2. Uses polygon filling to create precise masks following facial contours
3. Applies dilation to expand the mask slightly for better coverage
4. Uses a 7×7 kernel with 3 iterations for substantial expansion



7. **Tracking and Failure Handling**

```python
if not mask_updated:
    tracking_failure_count += 1
    if last_valid_mask is not None and tracking_failure_count <= max_tracking_failures:
        current_mask = last_valid_mask.copy()
```

1. If face detection fails, uses the last valid mask for up to 30 consecutive frames
2. Prevents the mask from disappearing during temporary detection failures
3. Resets the failure counter when face detection succeeds again



8. **Mask Smoothing**

```python
previous_masks.append(current_mask)
if len(previous_masks) > smoothing_frames:
    previous_masks.pop(0)

smoothed_mask = np.zeros_like(current_mask, dtype=np.float32)
for mask in previous_masks:
    smoothed_mask += mask.astype(np.float32)
smoothed_mask = (smoothed_mask / len(previous_masks)).astype(np.uint8)
```

1. Averages masks across 5 consecutive frames
2. Creates smoother transitions between frames
3. Prevents flickering or sudden mask changes
4. Converts the averaged float mask back to a binary mask



9. **Applying the Black Mask**

```python
black = np.zeros_like(frame)
masked_black = cv2.bitwise_and(black, black, mask=final_mask)
inverse_mask = cv2.bitwise_not(final_mask)
background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
frame = cv2.add(background, masked_black)
```

1. Creates a completely black frame
2. Uses the mask to combine the original frame with black regions
3. Only applies black to the masked areas (eyes, eyebrows, forehead)
4. Preserves the rest of the frame unchanged





## Complete Code

```python
import cv2
import dlib
import numpy as np

def process_video_with_improved_mask():
    # Hardcoded file names
    input_path = "test.mp4"
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
```

## Technical Details

### The 68-Point Facial Landmark System

The code uses dlib's 68-point facial landmark system where:

- Points 0-16: Jawline
- Points 17-21: Left eyebrow
- Points 22-26: Right eyebrow
- Points 27-35: Nose
- Points 36-41: Left eye
- Points 42-47: Right eye
- Points 48-67: Mouth and lips






### Key Components

#### 1. Region Expansion Function

```python
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
```

This function:

- Calculates the center point of a facial feature
- Expands the region by scaling each point outward from the center
- Ensures complete coverage of facial features
- Uses different expansion factors for different features:

- Eyes: 1.5 (50% larger)
- Eyebrows: 1.4 (40% larger)





#### 2. Mask Creation Process

The mask creation involves several steps:

1. **Polygon Creation**: Combines facial landmarks into a single polygon

```python
upper_face_points = [forehead_points[0]]
upper_face_points.append(forehead_points[1])
upper_face_points.extend(right_eyebrow[::-1])
# ... more points ...
cv2.fillPoly(current_mask, [np.array(upper_face_points)], 255)
```


2. **Individual Feature Enhancement**: Adds expanded regions for each feature

```python
cv2.fillPoly(current_mask, [np.array(expand_region(left_eye, 1.5))], 255)
cv2.fillPoly(current_mask, [np.array(expand_region(right_eye, 1.5))], 255)
```


3. **Forehead Coverage**: Adds a rectangle for the forehead region

```python
cv2.rectangle(current_mask, 
             (leftmost - 20, forehead_top), 
             (rightmost + 20, eyebrow_top), 
             255, -1)
```


4. **Dilation**: Expands the mask to ensure complete coverage

```python
kernel = np.ones((7, 7), np.uint8)
current_mask = cv2.dilate(current_mask, kernel, iterations=3)
```




#### 3. Tracking and Smoothing Mechanisms

The code implements two key mechanisms for consistent masking:

1. **Mask Tracking**:

```python
if not mask_updated:
    tracking_failure_count += 1
    if last_valid_mask is not None and tracking_failure_count <= max_tracking_failures:
        current_mask = last_valid_mask.copy()
```

1. Stores the last valid mask when face detection succeeds
2. Uses this stored mask when face detection fails
3. Maintains the mask for up to 30 consecutive frames of detection failure
4. Prevents the mask from disappearing during challenging video segments



2. **Temporal Smoothing**:

```python
previous_masks.append(current_mask)
if len(previous_masks) > smoothing_frames:
    previous_masks.pop(0)

smoothed_mask = np.zeros_like(current_mask, dtype=np.float32)
for mask in previous_masks:
    smoothed_mask += mask.astype(np.float32)
smoothed_mask = (smoothed_mask / len(previous_masks)).astype(np.uint8)
```

1. Maintains a sliding window of 5 recent masks
2. Averages these masks to create a smoothed version
3. Creates gradual transitions between frames
4. Prevents flickering or sudden mask changes
5. Converts the averaged float mask back to a binary mask using thresholding





#### 4. Mask Application

```python
black = np.zeros_like(frame)
masked_black = cv2.bitwise_and(black, black, mask=final_mask)
inverse_mask = cv2.bitwise_not(final_mask)
background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
frame = cv2.add(background, masked_black)
```

This process:

- Creates a completely black frame (`black`)
- Extracts only the masked regions from this black frame (`masked_black`)
- Creates an inverse mask for the original frame regions (`inverse_mask`)
- Extracts the non-masked regions from the original frame (`background`)
- Combines the black masked regions with the original background (`frame`)


## Troubleshooting

### Common Issues

#### 1. Face Detection Failures

**Symptoms**: Mask disappears for several frames

**Solutions**:

- Increase `max_tracking_failures` (default: 30) to maintain the mask longer during detection failures
- Adjust lighting in the original video if possible
- Increase face detector sensitivity with `detector(gray, 2)` (higher values increase sensitivity)
- Try different face detection approaches:

```python
# Alternative: Use Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
```




#### 2. Incomplete Eye Coverage

**Symptoms**: Parts of the eyes are still visible

**Solutions**:

- Increase the eye expansion factor:

```python
cv2.fillPoly(current_mask, [np.array(expand_region(left_eye, 1.7))], 255)  # Increase from 1.5 to 1.7
cv2.fillPoly(current_mask, [np.array(expand_region(right_eye, 1.7))], 255)
```


- Increase dilation:

```python
kernel = np.ones((9, 9), np.uint8)  # Increase from 7x7 to 9x9
current_mask = cv2.dilate(current_mask, kernel, iterations=4)  # Increase from 3 to 4
```


- Add additional padding around eye regions:

```python
# Add extra padding around eyes
left_eye_min_x = min([x for x, _ in left_eye]) - 10
left_eye_max_x = max([x for x, _ in left_eye]) + 10
left_eye_min_y = min([y for _, y in left_eye]) - 10
left_eye_max_y = max([y for _, y in left_eye]) + 10
cv2.rectangle(current_mask, (left_eye_min_x, left_eye_min_y), 
             (left_eye_max_x, left_eye_max_y), 255, -1)
```




#### 3. Video Codec Issues

**Symptoms**: Error when writing the output video

**Solutions**:

- Try different codecs:

```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative codec
# or
fourcc = cv2.VideoWriter_fourcc(*'H264')
```


- Change the output file extension to match the codec (e.g., .avi for XVID)
- For macOS:

```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # For macOS compatibility
```


- For complete compatibility:

```python
# Try to determine the best codec for the platform
import platform
if platform.system() == 'Darwin':  # macOS
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
elif platform.system() == 'Windows':
    fourcc = cv2.VideoWriter_fourcc(*'H264')
else:  # Linux and others
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
```




#### 4. Performance Issues

**Symptoms**: Processing is very slow

**Solutions**:

- Process at a lower resolution:

```python
# Resize frame for faster processing
scale_factor = 0.5  # Process at half resolution
small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

# Detect faces in smaller frame
faces = detector(small_gray)

# Scale coordinates back to original size
# ... process with scaled coordinates ...
```


- Skip frames for faster processing:

```python
# Process every nth frame
process_every_n = 2  # Process every 2nd frame
if frame_count % process_every_n != 0:
    # Copy mask from previous frame
    # ...
```




## Customization

### Adjusting the Mask Region

To modify which parts of the face are masked:

```python
# To mask only the eyes (not eyebrows or forehead)
# Comment out or remove these lines:
cv2.fillPoly(current_mask, [np.array(expand_region(left_eyebrow, 1.4))], 255)
cv2.fillPoly(current_mask, [np.array(expand_region(right_eyebrow, 1.4))], 255)
cv2.rectangle(current_mask, (leftmost - 20, forehead_top), (rightmost + 20, eyebrow_top), 255, -1)
```

### Changing Mask Color

To use a different color instead of black:

```python
# Replace this line:
black = np.zeros_like(frame)

# With a colored mask (e.g., red):
colored_mask = np.zeros_like(frame)
colored_mask[:, :, 2] = 255  # Red channel (BGR format)

# Then use colored_mask instead of black in subsequent code
masked_color = cv2.bitwise_and(colored_mask, colored_mask, mask=final_mask)
```

For a semi-transparent mask:

```python
# For a semi-transparent black mask (e.g., 80% black)
opacity = 0.8  # 0 = transparent, 1 = fully black
black = np.zeros_like(frame)

# Create a copy of the original frame
frame_copy = frame.copy()

# Apply the mask with transparency
cv2.addWeighted(black, opacity, frame_copy, 1-opacity, 0, frame_copy, mask=final_mask)

# Combine with the original frame
inverse_mask = cv2.bitwise_not(final_mask)
background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
frame = cv2.add(background, cv2.bitwise_and(frame_copy, frame_copy, mask=final_mask))
```

### Processing Multiple Videos

To process multiple videos, modify the script to accept command-line arguments:

```python
import argparse

def process_video_with_improved_mask(input_path, output_path):
    # ... same code but with parameterized input/output paths ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial anonymization tool')
    parser.add_argument('--input', default="test.mp4", help='Input video file')
    parser.add_argument('--output', default="output_blured.mp4", help='Output video file')
    parser.add_argument('--blur-strength', type=int, default=35, help='Blur strength (higher = stronger blur)')
    args = parser.parse_args()
    
    process_video_with_improved_mask(args.input, args.output)
```

### Adding a Preview Mode

To see the processing in real-time:

```python
# Add to argument parser
parser.add_argument('--preview', action='store_true', help='Show preview window')

# Then in the main loop after processing each frame
if args.preview:
    # Create a smaller preview window
    preview = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Preview', preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Performance Considerations

### Processing Time

- Face detection and landmark prediction are the most computationally intensive operations
- Processing time scales with:

- Video resolution
- Frame rate
- Number of faces in the frame
- CPU performance





### Memory Usage

- The script maintains several copies of each frame and mask in memory
- For high-resolution videos, memory usage can be significant
- The sliding window of previous masks adds additional memory overhead


### Optimization Strategies

1. **Reduce Processing Resolution**:

```python
# Process at lower resolution
scale = 0.5
small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
# ... process small_frame ...
# Scale mask back up before applying
final_mask = cv2.resize(final_mask, (width, height), interpolation=cv2.INTER_NEAREST)
```


2. **Skip Frames**:

```python
# Process every nth frame
if frame_count % 2 == 0:  # Process every other frame
    # ... perform detection and masking ...
else:
    # Copy mask from previous frame
    current_mask = last_valid_mask.copy()
```


3. **Parallel Processing**:

```python
import concurrent.futures

def process_frame(frame, frame_count):
    # ... process a single frame ...
    return processed_frame

# Process frames in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(0, total_frames, batch_size):
        batch_frames = frames[i:i+batch_size]
        future = executor.submit(process_batch, batch_frames, i)
        futures.append(future)
    
    for future in concurrent.futures.as_completed(futures):
        # ... handle results ...
```


4. **GPU Acceleration**:

1. Consider using GPU-accelerated libraries like CUDA-enabled OpenCV
2. For dlib, compile with CUDA support for faster face detection





---

This tool provides a robust solution for facial anonymization that maintains consistent coverage throughout the video, even during challenging sections where face detection might temporarily fail. The combination of precise facial landmark detection, region expansion, mask tracking, and temporal smoothing ensures high-quality results for privacy protection.

