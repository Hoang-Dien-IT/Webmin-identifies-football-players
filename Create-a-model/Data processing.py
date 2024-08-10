import cv2
import dlib
import os
from tqdm import tqdm
import numpy as np

def random_flip(image):
    flip_code = np.random.randint(1, 2)
    return cv2.flip(image, flip_code)
def random_rotate(image):
    rows, cols = image.shape[:2]
    angle = np.random.randint(-30, 30)
    matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows))
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ratio = 0.5 + np.random.uniform()
    hsv[:,:,2] = np.clip(hsv[:,:,2] * ratio, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def add_gaussian_noise(image, mean=0, stddev=0.4):
    noise = np.random.normal(mean, stddev, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

parent_directory_default = "Default-train-data/"
parent_directory_pretreatment = "Train_data/"

subdirectories = [subdir for subdir in os.listdir(parent_directory_default) if os.path.isdir(os.path.join(parent_directory_default, subdir))]
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(parent_directory_default, subdirectory)
    for file in os.listdir(subdirectory_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(subdirectory_path, file)
            image = cv2.imread(image_path)
            flipped_image = random_flip(image)
            rotated_image = random_rotate(image)
            brightened_image = random_brightness(image)
            noisy_image = add_gaussian_noise(image)
            flipped_rotated_image = random_rotate(flipped_image)
            flipped_rotated_brightened_image = random_brightness(flipped_rotated_image)
            rotated_rotated_brightened_noisy_image = add_gaussian_noise(flipped_rotated_brightened_image)

            cv2.imwrite(os.path.join(subdirectory_path, f'flipped_{file}'), flipped_image)
            cv2.imwrite(os.path.join(subdirectory_path, f'rotated_{file}'), rotated_image)
            cv2.imwrite(os.path.join(subdirectory_path, f'brightened_{file}'), brightened_image)
            cv2.imwrite(os.path.join(subdirectory_path, f'noisy_{file}'), noisy_image)
            cv2.imwrite(os.path.join(subdirectory_path, f'flipped_rotated_brightened_{file}'),
                        rotated_rotated_brightened_noisy_image)


def extract_faces_and_rename_images(input_folder, output_folder):
    desired_size = (100, 100)
    detector = dlib.get_frontal_face_detector()
    total_images = 0
    images_with_faces = 0
    images_without_faces = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for i, face in enumerate(faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cropped_face = image[y:y + h, x:x + w]
                if (cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0 or cropped_face.shape[2] == 0):
                    continue
                resized_face = cv2.resize(cropped_face, desired_size)
                output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{i:02d}.jpg')
                cv2.imwrite(output_path, resized_face)

            if len(faces) == 0:
                images_without_faces.append(filename)
            if len(faces) > 0:
                images_with_faces += 1

            total_images += 1

    percentage_with_faces = (images_with_faces / total_images) * 100
    info_str = f"Tổng số ảnh: {total_images}\nSố ảnh nhận diện được: {images_with_faces}\nTỉ Lệ nhận diện được: {percentage_with_faces:.2f}%\n"
    print(info_str)
    print("Quá trình trích xuất khuôn mặt đã hoàn tất.")

    if images_without_faces:
        print("\nCác ảnh không có khuôn mặt:")
        for img_name in images_without_faces:
            print(img_name)
def rename_images_in_subfolders(parent_dir):
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        if subdir.isdigit() and len(subdir) == 5:
            for i, filename in enumerate(sorted(os.listdir(subdir_path))):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    new_filename = f"{subdir}_{i:05d}.jpg"
                    os.rename(os.path.join(subdir_path, filename), os.path.join(subdir_path, new_filename))


subdirectories_1 = [subdir for subdir in os.listdir(parent_directory_default) if os.path.isdir(os.path.join(parent_directory_default, subdir))]
subdirectories_2 = [subdir for subdir in os.listdir(parent_directory_pretreatment) if os.path.isdir(os.path.join(parent_directory_pretreatment, subdir))]
with tqdm(total=1) as pbar:
    for subdirectory_1, subdirectory_2  in zip(subdirectories_1, subdirectories_2):
        subdirectory_path_default = os.path.join(parent_directory_default, subdirectory_1)
        subdirectory_path = os.path.join(parent_directory_pretreatment, subdirectory_2)
        input_folder = subdirectory_path_default
        output_folder = subdirectory_path
        extract_faces_and_rename_images(input_folder, output_folder)
        rename_images_in_subfolders(output_folder)
        pbar.update(1)
print("Hoàng Thành")


def rename_images_in_subdirectories(parent_dir):
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            for i, filename in enumerate(files):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    new_filename = f"{subdir}_{i:05}" + os.path.splitext(filename)[1]
                    old_filepath = os.path.join(subdir_path, filename)
                    new_filepath = os.path.join(subdir_path, new_filename)
                    os.rename(old_filepath, new_filepath)
                    print(f"Đổi tên '{filename}' thành '{new_filename}' trong thư mục '{subdir}'")
rename_images_in_subdirectories(parent_directory_pretreatment)
