# from ultralytics import YOLO
from ultralyticsplus import YOLO, render_result
import cv2
import cvzone
import math
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 PPE Detection")
    parser.add_argument('--image', type=str, help='Path to an image file')
    parser.add_argument('--video', type=str, help='Path to a video file (or leave blank for webcam)')
    parser.add_argument('--folder', type=str, help='Path to a folder containing images')
    args = parser.parse_args()

    model = YOLO("models/ppe.pt")
    # model = YOLO('keremberke/yolov8n-hard-hat-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    classNames = ['Helmet', 'Mask', 'Without_Helmet', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    #classNames = ['Helmet', 'Without_Helmet']

    myColor = (0, 0, 255)

    class_map = {
        "With Helmet": "Hardhat",
        "Without Helmet": "NO-Hardhat"
    }
    relevant_classes = {"Hardhat", "NO-Hardhat"}

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}")
            return
        results = model.predict(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if conf > 0.5:
                    if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0, 255)
                        print("Violation!!!")
                    elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                        myColor = (0, 255, 0)
                        print("Safe!!!")
                    else:
                        myColor = (255, 0, 0)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if args.folder:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [f for f in os.listdir(args.folder) if f.lower().endswith(image_extensions)]
        if not image_files:
            print(f"No image files found in folder: {args.folder}")
            return
        predictions = []
        for image_file in sorted(image_files):
            img_path = os.path.join(args.folder, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            results = model.predict(img)
            print(results)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    if classNames[cls]=='Helmet' or classNames[cls]=='Without_Helmet':
                        if conf > 0.25:
                            if currentClass == 'Without_Helmet' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                                myColor = (0, 0, 255)
                            elif currentClass == 'Helmet' or currentClass == 'Safety Vest' or currentClass == "Mask":
                                myColor = (0, 255, 0)
                            else:
                                myColor = (255, 0, 0)
                            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                                (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                                colorT=(255, 255, 255), colorR=myColor, offset=5)
                            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                            predictions.append({
                                'image': os.path.basename(img_path),
                                'class': currentClass,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'confidence': conf
                            })
            # cv2.imshow("Image", img)
            # print(f"Showing: {img_path} (press any key for next, or ESC to quit)")
            # key = cv2.waitKey(0)
            # if key == 27:  # ESC key
            #     break
        cv2.destroyAllWindows()
        # Save all predictions to a CSV
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv('/Users/adil_zhiyenbayev/adil_code/helmet_detection/Safety-Detection-YOLOv8/images/archive/labels/predictions_test_resided640.csv', index=False)
        return

    # Video or webcam
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return
    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if conf > 0.5:
                    if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0, 255)
                        print("Violation!!!")
                    elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                        myColor = (0, 255, 0)
                        print("Safe!!!")
                    else:
                        myColor = (255, 0, 0)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
