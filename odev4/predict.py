from ultralytics import YOLO
import cv2
import os
import sys

# Model path
model_path = 'runs/detect/apricot_model/weights/best.pt'

# Load the trained YOLO model
model = YOLO(model_path)


def predict_image(image_path, save_dir='predictions'):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Perform prediction
    results = model(image)

    # Save prediction results
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, os.path.basename(image_path))
    annotated_image = results[0].plot()  # Annotate image with bounding boxes and labels
    cv2.imwrite(output_path, annotated_image)
    print(f"Prediction saved to {output_path}")


def predict_video(video_path, save_dir='predictions', display=False):
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Define output video settings
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"pred_{os.path.basename(video_path)}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on each frame
        results = model(frame)
        annotated_frame = results[0].plot()  # Annotate frame

        # Write frame to output video
        out.write(annotated_frame)

        # Display frame if required
        if display:
            cv2.imshow('Prediction', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Prediction video saved to {output_path}")


if __name__ == "__main__":
    # Check if input path is provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_video_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if input is an image or video
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            predict_image(input_path)
        elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            predict_video(input_path, display=True)
        else:
            print("Error: Unsupported file format.")
    else:
        print("Error: File not found.")
