"""
YOLOv8-based Person Detection System
Using Ultralytics YOLOv8 for enhanced performance
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8PersonDetector:
    """
    Modern person detection using YOLOv8 from Ultralytics
    Simplified and more efficient implementation
    """
    
    def __init__(self, 
                 model_size: str = 'n',
                 confidence_threshold: float = 0.5,
                 device: str = 'cuda:0'):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detection
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load YOLOv8 model
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading {model_name}...")
        
        try:
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.warning(f"Failed to load on {device}, falling back to CPU: {e}")
            self.device = 'cpu'
            self.model = YOLO(model_name)
            self.model.to('cpu')
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def calculate_fps(self) -> float:
        """Calculate frames per second"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
            
        return self.fps
    
    def detect_and_draw(self, frame: np.ndarray) -> tuple:
        """
        Detect persons and draw bounding boxes
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, person_count)
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, classes=[0], verbose=False)
        
        # Get detections for person class (0)
        person_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f'Person: {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Background rectangle for text
                cv2.rectangle(
                    frame,
                    (x1, y1 - 25),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Put text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
                
                person_count += 1
        
        return frame, person_count
    
    def add_overlay(self, frame: np.ndarray, person_count: int) -> np.ndarray:
        """Add information overlay"""
        fps = self.calculate_fps()
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add info text
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: YOLOv8{self.model_size}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self, source: int = 0, show: bool = True, save: Optional[str] = None):
        """
        Run real-time detection
        
        Args:
            source: Camera index or video file path
            show: Display output window
            save: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error("Cannot open camera/video")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if saving
        writer = None
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save, fourcc, fps_video, (width, height))
            logger.info(f"Saving output to {save}")
        
        logger.info("Starting detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and annotate
                frame, person_count = self.detect_and_draw(frame)
                frame = self.add_overlay(frame, person_count)
                
                # Save frame if requested
                if writer:
                    writer.write(frame)
                
                # Display
                if show:
                    cv2.imshow('YOLOv8 Person Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info("Detection stopped")


def main():
    """Main execution function"""
    # Initialize detector with YOLOv8 nano (fastest)
    detector = YOLOv8PersonDetector(
        model_size='n',  # Options: 'n', 's', 'm', 'l', 'x'
        confidence_threshold=0.5,
        device='cuda:0'  # Use 'cpu' if no GPU
    )
    
    # Run detection on webcam
    detector.run(source=0, show=True, save=None)


if __name__ == "__main__":
    main()
