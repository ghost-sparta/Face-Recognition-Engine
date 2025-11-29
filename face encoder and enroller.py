import face_recognition
import pickle
import os
import cv2

def create_face_encodings(images_dir="../know_faces", output_file="../data/face_encodings.pkl"):
    """
    Create face encodings from known face images.
    Folder structure: known_faces/person_name/image1.jpg, image2.jpg, etc.
    """
    known_face_encodings = []
    known_face_names = []
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    for person_name in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Processing {person_name}...")
        
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Use the first face found in the image
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
                print(f"  Added encoding from {image_file}")
            else:
                print(f"  No face found in {image_file}")
    
    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
    print(f"Saved {len(known_face_encodings)} face encodings to {output_file}")
    return known_face_encodings, known_face_names

def enroll_face_from_camera(name, encodings_file="data/face_encodings.pkl"):
    """Capture face from camera and add to encodings."""
    # Load existing encodings
    try:
        with open(encodings_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
    except FileNotFoundError:
        known_face_encodings, known_face_names = [], []
    
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture face, ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display frame
        cv2.imshow('Enroll Face - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE to capture
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame)
            
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                
                # Save updated encodings
                with open(encodings_file, 'wb') as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                
                print(f"Successfully enrolled {name}")
                break
            else:
                print("No face detected. Try again.")
        
        elif key == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    #create_face_encodings("../../known_faces", "../data/face_encodings.pkl")
    enroll_face_from_camera('jane', encodings_file="/data/face_encodings.pkl")