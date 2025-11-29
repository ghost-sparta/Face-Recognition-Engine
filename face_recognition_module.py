"""
Optimized Face Recognition Module for Production
Features: Performance optimizations, caching, error handling, and scalability
"""

import cv2
import face_recognition
import pickle
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Data class to store recognition results"""
    name: str
    confidence: float
    face_location: Tuple[int, int, int, int]
    timestamp: float


class FaceRecognition:
    """
    Optimized face recognition system with performance enhancements
    for production environments.
    """

    def __init__(self,
                 encodings_file: str = "data/face_encodings.pkl",
                 confidence_threshold: float = 0.6,
                 processing_scale: float = 0.5,
                 process_every_n_frames: int = 3,
                 max_workers: int = 2):
        """
        Initialize the face recognition system.

        Args:
            encodings_file: Path to saved face encodings
            confidence_threshold: Minimum confidence for positive recognition
            processing_scale: Scale factor for processing frames (0.0-1.0)
            process_every_n_frames: Process every nth frame for performance
            max_workers: Number of worker threads for parallel processing
        """
        self.encodings_file = encodings_file
        self.confidence_threshold = confidence_threshold
        self.processing_scale = processing_scale
        self.process_every_n_frames = process_every_n_frames
        self.max_workers = max_workers

        # Performance optimizations
        self.frame_counter = 0
        self.last_processed_frame = None
        self.last_results: List[RecognitionResult] = []
        self.processing_lock = threading.Lock()

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'processing_time': 0,
            'last_processing_time': 0
        }

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

        logger.info(f"FaceRecognition initialized with {len(self.known_face_names)} known faces")

    # In face_recognition_module.py - FIX THE ENCODING LOADING
    def load_encodings(self) -> bool:
        """
        Load face encodings from file with proper error handling.
        """
        try:
            if not os.path.exists(self.encodings_file):
                logger.warning(f"Encodings file not found: {self.encodings_file}")
                # Create empty encodings structure
                self.known_face_encodings = []
                self.known_face_names = []
                return True

            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)

            # Handle different data formats
            if isinstance(data, dict):
                # New format: {'encodings': [], 'names': []}
                self.known_face_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
            elif isinstance(data, tuple) and len(data) == 2:
                # Old format: (encodings, names)
                self.known_face_encodings, self.known_face_names = data
            elif isinstance(data, list):
                # Assume it's just encodings list
                self.known_face_encodings = data
                self.known_face_names = [f"Person_{i}" for i in range(len(data))]
            else:
                logger.error(f"Unknown encodings file format: {type(data)}")
                self.known_face_encodings = []
                self.known_face_names = []

            logger.info(f"Loaded {len(self.known_face_names)} face encodings")
            return True

        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            # Initialize empty to avoid crashes
            self.known_face_encodings = []
            self.known_face_names = []
            return False

    def save_encodings(self) -> bool:
        """
        Save current face encodings to file.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.encodings_file), exist_ok=True)

            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }

            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved {len(self.known_face_names)} face encodings to {self.encodings_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving encodings: {e}")
            return False

    def add_face_encoding(self, image: np.ndarray, name: str) -> bool:
        """
        Add a new face encoding to the known faces database.

        Args:
            image: RGB image array
            name: Name associated with the face

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                logger.warning(f"No face found in image for {name}")
                return False

            # Use the first face found
            encoding = face_encodings[0]

            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

            logger.info(f"Added face encoding for {name}")
            return True

        except Exception as e:
            logger.error(f"Error adding face encoding for {name}: {e}")
            return False

    def recognize_faces(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimized face recognition with frame skipping and caching.

        Args:
            frame: Input BGR frame from camera

        Returns:
            np.ndarray: Annotated frame with recognition results
        """
        start_time = time.time()
        self.frame_counter += 1

        # Only process every nth frame for performance
        if self.frame_counter % self.process_every_n_frames != 0:
            if self.last_results:
                return self.apply_cached_results(frame.copy())
            return frame

        try:
            # Use smaller frame for processing
            small_frame = cv2.resize(frame, (0, 0), fx=self.processing_scale, fy=self.processing_scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process recognitions
            current_results = []
            names = []

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                name, confidence = self._recognize_face(face_encoding)
                names.append(name)

                # Store result
                result = RecognitionResult(
                    name=name,
                    confidence=confidence,
                    face_location=face_location,
                    timestamp=time.time()
                )
                current_results.append(result)

            # Update cached results
            with self.processing_lock:
                self.last_results = current_results
                self.last_processed_frame = frame.copy()

            # Update statistics
            processing_time = time.time() - start_time
            self.stats['last_processing_time'] = processing_time
            self.stats['processing_time'] += processing_time
            self.stats['frames_processed'] += 1
            self.stats['faces_detected'] += len(face_locations)

            # Apply results to frame
            return self.apply_recognition_results(frame, names, face_locations)

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return frame

    def _recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a single face encoding.

        Args:
            face_encoding: Face encoding to recognize

        Returns:
            Tuple[str, float]: (name, confidence)
        """
        if not self.known_face_encodings:
            return "Unknown", 0.0

        # Calculate face distances
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

        if len(face_distances) == 0:
            return "Unknown", 0.0

        # Find best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0.0, 1.0 - best_distance)

        if confidence >= self.confidence_threshold:
            return self.known_face_names[best_match_index], confidence
        else:
            return "Unknown", confidence

    def apply_recognition_results(self,
                                  frame: np.ndarray,
                                  names: List[str],
                                  face_locations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Apply recognition results to the frame.

        Args:
            frame: Original frame
            names: List of recognized names
            face_locations: List of face locations

        Returns:
            np.ndarray: Annotated frame
        """
        display_frame = frame.copy()
        scale_factor = int(1 / self.processing_scale)

        for name, (top, right, bottom, left) in zip(names, face_locations):
            # Scale back to original size
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor

            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

            # Draw label background
            label = name if name != "Unknown" else "Unknown Person"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame,
                          (left, bottom - label_size[1] - 10),
                          (left + label_size[0], bottom),
                          color,
                          -1)

            # Draw label text
            cv2.putText(display_frame, label, (left, bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add performance stats to frame
        self._add_performance_stats(display_frame)

        return display_frame

    def apply_cached_results(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply cached recognition results to frame for intermediate frames.

        Args:
            frame: Input frame

        Returns:
            np.ndarray: Frame with cached results applied
        """
        display_frame = frame.copy()
        scale_factor = int(1 / self.processing_scale)

        with self.processing_lock:
            for result in self.last_results:
                top, right, bottom, left = result.face_location

                # Scale back to original size
                top *= scale_factor
                right *= scale_factor
                bottom *= scale_factor
                left *= scale_factor

                # Draw bounding box
                color = (0, 255, 0) if result.name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

                # Draw label with confidence
                label = f"{result.name} ({result.confidence:.2f})" if result.name != "Unknown" else "Unknown Person"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame,
                              (left, bottom - label_size[1] - 10),
                              (left + label_size[0], bottom),
                              color,
                              -1)

                cv2.putText(display_frame, label, (left, bottom - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add performance stats to frame
        self._add_performance_stats(display_frame)

        return display_frame

    def _add_performance_stats(self, frame: np.ndarray) -> None:
        """Add performance statistics to the frame."""
        stats_text = [
            f"FPS: {1 / self.stats['last_processing_time']:.1f}" if self.stats[
                                                                        'last_processing_time'] > 0 else "FPS: Calculating",
            f"Faces: {len(self.last_results)}",
            f"Frame: {self.frame_counter}",
            f"Processed: {self.stats['frames_processed']}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get current performance statistics.

        Returns:
            Dict[str, float]: Statistics dictionary
        """
        avg_processing_time = (self.stats['processing_time'] /
                               self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0)

        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'known_faces_count': len(self.known_face_names),
            'current_frame': self.frame_counter
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'processing_time': 0,
            'last_processing_time': 0
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.thread_pool.shutdown(wait=True)
        logger.info("FaceRecognition cleanup completed")


# Example usage and testing
def main():
    """Example usage of the FaceRecognition class."""
    import argparse

    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--encodings', type=str, default='data/face_encodings.pkl',
                        help='Face encodings file')
    args = parser.parse_args()

    # Initialize face recognition
    face_recog = FaceRecognition(
        encodings_file=args.encodings,
        confidence_threshold=0.6,
        processing_scale=0.5,
        process_every_n_frames=3
    )

    # Start camera
    cap = cv2.VideoCapture(args.camera)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = face_recog.recognize_faces(frame)

            # Display result
            cv2.imshow('Face Recognition', processed_frame)

            # Print statistics occasionally
            if face_recog.frame_counter % 30 == 0:
                stats = face_recog.get_statistics()
                logger.info(f"Stats: {stats}")

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_recog.cleanup()

        # Print final statistics
        stats = face_recog.get_statistics()
        logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    main()