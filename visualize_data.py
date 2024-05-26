import json
import os
import cv2
import random

json_file_path = r"C:\Users\LAPTOP\Traffic_Sign_Detection\via-trafficsign-coco-20210321\annotations\val.json"
images_folder = r"C:\Users\LAPTOP\Traffic_Sign_Detection\via-trafficsign-coco-20210321\val"

with open(json_file_path, 'r') as f:
    data = json.load(f)

random_images = random.sample(data['images'], 50)

for image_info in random_images:
    image_id = image_info['id']
    image_filename = image_info['file_name']
    image_path = os.path.join(images_folder, image_filename)
    img_annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_filename}. Skipping...")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image Info:\nID: {image_id}\nFile Name: {image_filename}\nWidth: {image_info['width']}\nHeight: {image_info['height']}\n")
    
    yolo_annotations = []
    
    # Vẽ bounding box trên ảnh sử dụng annotations
    for ann in img_annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == category_id), 'Unknown')
        # Tính thông tin YOLO
        x_center = x + w / 2
        y_center = y + h / 2
        yolo_bbox = [category_id, x_center / image_info['width'], y_center / image_info['height'], w / image_info['width'], h / image_info['height']]
        yolo_annotations.append(yolo_bbox)
        print(f"Annotation:\nCategory: {category_name}\nBounding Box: {bbox}\n")
        # Vẽ bounding box và category trên ảnh
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        cv2.putText(image, category_name, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # In thông tin YOLO
    print("\nYOLO Annotations:")
    for yolo_ann in yolo_annotations:
        print(" ".join(map(str, yolo_ann)))
    
    cv2.imshow(f"Image ID: {image_id}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ASCII của phím Enter
            break
    cv2.destroyAllWindows()
