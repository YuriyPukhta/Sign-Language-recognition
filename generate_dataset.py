import os
import cv2
import numpy as np
import random
import uuid

images_per_sign = 200
caltech_path = "./caltech_flattened"
asl_path = "./asl_dataset"
asl_dst = "./asl_augmented"

if not os.path.exists(asl_dst):
    os.makedirs(asl_dst)

asl_dirs = [x[0] for x in os.walk(asl_path)][1:]
caltech_images = os.listdir(caltech_path)

for subdir in asl_dirs:
    for _ in range(images_per_sign):
        # select images
        asl_images = os.listdir(subdir)
        asl_img_name = np.random.choice(asl_images)
        # asl_img_name = 'hand1_a_top_seg_4_cropped.jpeg'
        asl_img = cv2.imread(os.path.join(subdir, asl_img_name))
        caltech_image_name = np.random.choice(caltech_images)
        caltech_image = cv2.imread(os.path.join(caltech_path, caltech_image_name))
        caltech_image = cv2.resize(caltech_image, asl_img.shape[:2])

        # resize hand image
        img_ratio = random.uniform(0.5, 0.95)
        height, width = asl_img.shape[:2]
        resized_width = int(width * img_ratio)
        resized_height = int(height * img_ratio)

        margin_width = width - resized_width
        margin_height = height - resized_height

        top_margin = margin_height // 2
        bottom_margin = margin_height - top_margin
        left_margin = margin_width // 2
        right_margin = margin_width - left_margin

        resized_image = cv2.resize(asl_img, (resized_width, resized_height))
        processed_image = np.zeros((height, width, 3), dtype=np.uint8)
        processed_image[top_margin:top_margin + resized_height, left_margin:left_margin + resized_width] = resized_image

        # rotate
        angle = random.randint(-60, 60)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        processed_image = cv2.warpAffine(processed_image, rotation_matrix, (width, height))

        # shift
        max_x_shift = int((width - width * img_ratio) / 2)
        max_y_shift = int((height - height * img_ratio) / 2)
        shift_x = random.randint(-max_x_shift, max_x_shift)
        shift_y = random.randint(-max_y_shift, max_y_shift)

        canvas = np.zeros_like(processed_image)

        start_x = max(0, shift_x)
        start_y = max(0, shift_y)
        end_x = min(width, width + shift_x)
        end_y = min(height, height + shift_y)

        roi_start_x = max(0, -shift_x)
        roi_start_y = max(0, -shift_y)
        roi_end_x = min(width, width - shift_x)
        roi_end_y = min(height, height - shift_y)

        canvas[start_y:end_y, start_x:end_x] = processed_image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        processed_image = canvas

        # clean black noise around hand
        pixel_means = np.mean(processed_image, axis=2)
        mask = pixel_means < 10

        # remove small masked areas
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                cv2.drawContours(mask, [contour], 0, 0, -1)

        # remove noise
        cv2.dilate(mask, np.ones((5, 5)), mask)
        cv2.erode(mask, np.ones((5, 5)), mask)
        # remove black outline
        cv2.dilate(mask, np.ones((3, 3)), mask)

        # mask image
        processed_image = np.where(mask[:, :, np.newaxis], [0, 0, 0], processed_image).astype(np.uint8)

        # merge hand with background
        merged_image = np.where(mask[:, :, np.newaxis], caltech_image, processed_image).astype(np.uint8)

        # blur edge
        kernel_size = 3
        edges = cv2.Canny(mask * 255, 50, 150)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        blurred_image = cv2.filter2D(merged_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        edges = cv2.filter2D(edges, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        merged_image[edges != 0] = blurred_image[edges != 0]

        # save image
        asl_dst_path = os.path.join(asl_dst, os.path.split(subdir)[-1])
        if not os.path.exists(asl_dst_path):
            os.makedirs(asl_dst_path)
        img_name = str(uuid.uuid4()) + ".png"
        cv2.imwrite(os.path.join(asl_dst_path, img_name), merged_image)
