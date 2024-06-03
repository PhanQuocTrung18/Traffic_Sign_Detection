import os
import cv2
import random

images_folder = r"C:\Users\LAPTOP\Traffic_Sign_Detection\Data Converted\train\images"
labels_folder = r"C:\Users\LAPTOP\Traffic_Sign_Detection\Data Converted\train\labels"

category_names = {
    0: "stop",
    1: "left",
    2: "right",
    3: "straight",
    4: "no_left",
    5: "no_right"
}

label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
random_label_files = random.sample(label_files, 50)

for label_file in random_label_files:
    image_filename = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(images_folder, image_filename)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_filename}. Skipping...")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image File: {image_filename}\n")
    
    label_file_path = os.path.join(labels_folder, label_file)
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label_parts = line.strip().split()
            category_id = int(label_parts[0])
            x_center, y_center, w, h = map(float, label_parts[1:])
            image_height, image_width, _ = image.shape
            
            x = int((x_center - w / 2) * image_width)
            y = int((y_center - h / 2) * image_height)
            w = int(w * image_width)
            h = int(h * image_height)
            
            category_name = category_names.get(category_id, 'Unknown')
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, f"{category_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("Image with Bounding Boxes", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
