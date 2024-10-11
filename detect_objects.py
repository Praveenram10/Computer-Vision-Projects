import cv2
import argparse
from ultralytics import YOLO

def detect_objects(model_path, input_path, output_path):
    model = YOLO(model_path)

    if input_path.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(input_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for result in results:
                boxes = result.boxes  # Get detections
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    class_id = int(box.cls[0])  # Class ID

                    label = f'{model.names[class_id]}: {confidence:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    else:
        image = cv2.imread(input_path)
        results = model(image)


        for result in results:
            boxes = result.boxes  # Get detections
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                # Draw bounding box
                label = f'{model.names[class_id]}: {confidence:.2f}'
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument(
        "--model_path",  # Path to the YOLOv8 model
        type=str,
        help='Path to the YOLOv8 model file (e.g., yolov8n.pt)',
        required=True
    )
    parser.add_argument(
        "--input_path",  # Path to the input image or video
        type=str,
        help='Path to the input image or video file',
        required=True
    )
    parser.add_argument(
        "--output_path",  # Path to save the output
        type=str,
        help='Path to save the output image or video',
        required=True
    )

    args = parser.parse_args()
    detect_objects(args.model_path, args.input_path, args.output_path)

