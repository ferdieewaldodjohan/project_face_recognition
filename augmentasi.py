import cv2
import numpy as np
import os
import random

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale_factor):
    scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled

def translate_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return translated

def flip_image(image, flip_code):
    flipped = cv2.flip(image, flip_code)
    return flipped

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 400
        else:
            shadow = 0
            highlight = 400 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 200 * (contrast + 150) / (150 * (200 - contrast))
        alpha_c = f
        gamma_c = 150 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, gauss)
    return noisy

def crop_image(image, x, y, width, height):
    cropped = image[y:y+height, x:x+width]
    return cropped

def perspective_transform(image, src_points, dst_points):
    h, w = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, M, (w, h))
    return transformed

def color_jitter(image, hue_delta=18, saturation_scale=1.5, brightness_scale=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)
    hue = np.clip(hue, 0, 179).astype(np.uint8)
    hsv[:, :, 0] = hue
    saturation = np.clip(hsv[:, :, 1] * random.uniform(1/saturation_scale, saturation_scale), 0, 255).astype(np.uint8)
    hsv[:, :, 1] = saturation
    brightness = np.clip(hsv[:, :, 2] * random.uniform(1/brightness_scale, brightness_scale), 0, 255).astype(np.uint8)
    hsv[:, :, 2] = brightness
    jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return jittered

def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Rotate
    augmented_images.append(rotate_image(image, 90))
    augmented_images.append(rotate_image(image, -90))

    # Scale
    augmented_images.append(scale_image(image, 2))
    augmented_images.append(scale_image(image, 1))

    # Translate
    augmented_images.append(translate_image(image, 20, 0))
    augmented_images.append(translate_image(image, -20, 0))
    augmented_images.append(translate_image(image, 0, 20))
    augmented_images.append(translate_image(image, 0, -20))

    # Flip
    augmented_images.append(flip_image(image, 1))  # Horizontal flip
    augmented_images.append(flip_image(image, 0))  # Vertical flip

    # Brightness and Contrast
    augmented_images.append(adjust_brightness_contrast(image, 30, 30))
    augmented_images.append(adjust_brightness_contrast(image, -80, -80))

    # Gaussian Noise
    augmented_images.append(add_gaussian_noise(image, 0, 25))

    # Crop
    h, w = image.shape[:2]
    augmented_images.append(crop_image(image, 20, 20, w-40, h-40))

    # Perspective Transform
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points[1] += (20, 0)
    dst_points[2] += (0, 20)
    augmented_images.append(perspective_transform(image, src_points, dst_points))

    # Color Jitter
    augmented_images.append(color_jitter(image))

    # Black and White
    augmented_images.append(convert_to_grayscale(image))

    return augmented_images

known_directory = "known_faces"
output_directory = "augmented_faces"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each photo in the known directory
for folder in os.listdir(known_directory):
    input_folder_path = os.path.join(known_directory, folder)
    output_folder_path = os.path.join(output_directory, folder)
    
    # Ensure subfolder exists in augmented_faces
    os.makedirs(output_folder_path, exist_ok=True)
    
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder_path, filename)
            image = cv2.imread(image_path)

            # Augment the image
            augmented_images = augment_image(image)

            # Save augmented images in the corresponding folder
            for i, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
                cv2.imwrite(output_path, augmented_image)
