import os
import random
import shutil

def create_train_test_split(source_dir, train_dir, test_dir, sample_ratio=0.2):
    classes = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for cls in classes:
        cls_source_dir = os.path.join(source_dir, cls)
        cls_train_dir = os.path.join(train_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)
        
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_test_dir, exist_ok=True)
        
        images = os.listdir(cls_source_dir)
        num_samples = int(len(images) * sample_ratio)
        sampled_images = random.sample(images, num_samples)
        
        for img in sampled_images:
            src_img_path = os.path.join(cls_source_dir, img)
            dst_img_path = os.path.join(cls_test_dir, img)
            shutil.move(src_img_path, dst_img_path)
        
        for img in os.listdir(cls_source_dir):
            src_img_path = os.path.join(cls_source_dir, img)
            dst_img_path = os.path.join(cls_train_dir, img)
            shutil.move(src_img_path, dst_img_path)

source_dir = r"G:\내 드라이브\2024 Summer\code\project_sign\dataset\train"
train_dir = r"G:\내 드라이브\2024 Summer\code\project_sign\dataset\train"
test_dir = r"G:\내 드라이브\2024 Summer\code\project_sign\dataset\test"

# 샘플링 비율 설정 (예: 20%)
sample_ratio = 0.15

create_train_test_split(source_dir, train_dir, test_dir, sample_ratio)