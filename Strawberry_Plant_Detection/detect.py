from ultralytics import YOLO
from PIL import Image
import cv2

def detect(input_image_name):

    # Absolute path for model weights on Mac 14 pro 
    model = YOLO("/Users/alex/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Documents/Georgia_Tech/GTRI FarmHand/Code/IndoorFarming/Strawberry_Plant_Detection/runs/detect/train20/weights/best.pt")
    results = model.predict(source=input_image_name, conf=0.9)

    num_flowers = len(results[0])
    print("Number of flowers detected: ", num_flowers)

    # for result in results:

    #     result.plot(conf=True, boxes=True, show=True)

    # Process each result
    for result in results:
        # Access the Boxes object
        boxes = result.boxes
        # Check if there are any detections and boxes.xyxy is not empty
        if boxes is not None and len(boxes.xyxy) > 0:
            # Load the image once, to draw all detected boxes
            frame = cv2.imread(input_image_name)
            # Iterate through each detected box
            for box, confidence in zip(boxes.xyxy, boxes.conf):
                # Drawing the rectangle using coordinates from box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (x1 - 100, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Save the annotated image after drawing all boxes
            cv2.imwrite('test1_YOLO_output.png', frame)

    return boxes