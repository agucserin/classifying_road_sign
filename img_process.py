import os
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np

def parse_annotation(ann_dir, img_dir, output_dir):
    for ann in os.listdir(ann_dir):
        if ann.endswith('.xml'):
            tree = ET.parse(os.path.join(ann_dir, ann))
            root = tree.getroot()
            img_name = root.find('filename').text
            img_path = os.path.join(img_dir, img_name)
            
            image = Image.open(img_path)
            image = image.convert("RGB")
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for obj in root.findall('object'):
                cls = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                face = img_cv[ymin:ymax, xmin:xmax]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                cls_dir = os.path.join(output_dir, cls)
                if not os.path.exists(cls_dir):
                    os.makedirs(cls_dir)
                
                face_filename = f"{os.path.splitext(img_name)[0]}_{xmin}_{ymin}.jpg"
                face_path = os.path.join(cls_dir, face_filename)
                face_pil.save(face_path)

ann_dir = r"C:\Users\MSI\Downloads\road_sign\annotations"
img_dir = r"C:\Users\MSI\Downloads\road_sign\images"
output_dir = r"C:\Users\MSI\Downloads\road_sign\output"

parse_annotation(ann_dir, img_dir, output_dir)