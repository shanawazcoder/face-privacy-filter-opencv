import cv2
import numpy as np
import argparse
import sys
import os

class FacePrivacyFilter:
    def __init__(self, mode='pixelate', strength=20):
        """
        Initialize the Face Privacy Filter application.
        
        Args:
            mode (str): Initial filter mode ('pixelate' or 'blur')
            strength (int): Initial strength value for the filter
        """
        self.mode = mode
        self.strength = strength
        self.running = True
        
        # Try to load the Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        
        # Check if the file exists
        if not os.path.exists(cascade_path):
           
            alt_paths = [
                'haarcascade_frontalface_alt2.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
            else:
                raise RuntimeError("Failed to locate Haar Cascade classifier file")
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        print(f"Successfully loaded Haar Cascade classifier from: {cascade_path}")
    
    def pixelate_region(self, roi, pixel_size):
        """
        Apply pixelation effect to a region of interest.
        
        Args:
            roi (numpy.ndarray): Region of interest to pixelate
            pixel_size (int): Size of pixels (smaller values = more pixelated)
            
        Returns:
            numpy.ndarray: Pixelated region
        """
        if pixel_size < 1:
            pixel_size = 1
            
        # Get original dimensions
        h, w = roi.shape[:2]
        
        # Resize down
        temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # Resize back to original size
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def blur_region(self, roi, kernel_size):
        """
        Apply Gaussian blur to a region of interest.
        
        Args:
            roi (numpy.ndarray): Region of interest to blur
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            
        Returns:
            numpy.ndarray: Blurred region
        """
       
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
            
    
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        
        return blurred
    
    def is_valid_face(self, x, y, w, h, frame_shape):
        """
        Check if a detected face is likely to be valid based on its properties.
        
        Args:
            x, y, w, h: Face bounding box coordinates and dimensions
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            bool: True if the face is likely valid, False otherwise
        """
        # Calculate aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        # Check if the face is too small relative to the frame
        frame_h, frame_w = frame_shape[:2]
        min_face_size = min(frame_h, frame_w) * 0.05
        if w < min_face_size or h < min_face_size:
            return False
            
        # Check if the face is too close to the edge of the frame
        edge_threshold = 0.05 
        if (x < frame_w * edge_threshold or 
            y < frame_h * edge_threshold or
            x + w > frame_w * (1 - edge_threshold) or
            y + h > frame_h * (1 - edge_threshold)):
            return False
            
        return True
    
    def process_frame(self, frame):
        """
        Process a single video frame to detect and filter faces.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            numpy.ndarray: Processed frame with filtered faces
        """
      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    
            minNeighbors=5,  
            minSize=(50, 50), 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter out likely false positives
        valid_faces = []
        for (x, y, w, h) in faces:
            if self.is_valid_face(x, y, w, h, frame.shape):
                valid_faces.append((x, y, w, h))
        
        
        if len(valid_faces) > 0:
            print(f"Detected {len(valid_faces)} valid faces")
        
        
        for (x, y, w, h) in valid_faces:
            face_roi = frame[y:y+h, x:x+w]
            
            if self.mode == 'pixelate':
                pixel_size = max(1, min(w, h) // self.strength)
                filtered_roi = self.pixelate_region(face_roi, pixel_size)
            else: 
                kernel_size = self.strength * 2 + 1
                filtered_roi = self.blur_region(face_roi, kernel_size)
            
            frame[y:y+h, x:x+w] = filtered_roi
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    
    def draw_ui(self, frame):
        """
        Draw UI elements on the frame showing current settings.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on
            
        Returns:
            numpy.ndarray: Frame with UI elements
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        mode_text = f"Mode: {self.mode.upper()}"
        strength_text = f"Strength: {self.strength}"
        
        cv2.putText(frame, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, strength_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        instructions = "P: Toggle Mode | +/-: Adjust Strength | Q: Quit"
        cv2.putText(frame, instructions, (frame.shape[1] - 450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def handle_keypress(self, key):
        """
        Handle keyboard input for mode switching and strength adjustment.
        
        Args:
            key (int): ASCII value of the pressed key
        """
        # Toggle between pixelate and blur modes
        if key == ord('p') or key == ord('P'):
            self.mode = 'blur' if self.mode == 'pixelate' else 'pixelate'
            print(f"Switched to {self.mode} mode")
        
        # Increase strength
        elif key == ord('+') or key == ord('='):
            self.strength = min(50, self.strength + 1)
            print(f"Strength increased to {self.strength}")
        
        # Decrease strength
        elif key == ord('-') or key == ord('_'):
            self.strength = max(1, self.strength - 1)
            print(f"Strength decreased to {self.strength}")
        
        # Quit application
        elif key == ord('q') or key == ord('Q'):
            self.running = False
            print("Quitting application...")
    
    def run(self):
        """
        Main application loop that captures and processes video.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        # Set frame size for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Face Privacy Filter started")
        print("Press 'P' to toggle between pixelate/blur modes")
        print("Press '+/-' to adjust filter strength")
        print("Press 'Q' to quit")
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                processed_frame = self.process_frame(frame)
                
                processed_frame = self.draw_ui(processed_frame)
                
                cv2.imshow('Face Privacy Filter', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key != 255: 
                    self.handle_keypress(key)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

def main():
    """
    Parse command line arguments and run the application.
    """
    parser = argparse.ArgumentParser(
        description='Real-time face privacy filter with pixelation and blur effects',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['pixelate', 'blur'],
        default='pixelate',
        help='Initial filter mode'
    )
    
    parser.add_argument(
        '--strength',
        type=int,
        default=20,
        help='Initial filter strength (1-50)'
    )
    
    args = parser.parse_args()
    
    if not 1 <= args.strength <= 50:
        print("Warning: Strength should be between 1 and 50. Using default value.")
        args.strength = 20
    
    try:
        app = FacePrivacyFilter(mode=args.mode, strength=args.strength)
        app.run()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
