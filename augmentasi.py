import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import dlib

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

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

# def convert_to_grayscale(image):
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return grayscale

def perspective_rotate_y(image, angle):
    """
    Rotate the image around the Y-axis (perspective view).
    """
    h, w = image.shape[:2]
    f = max(h, w)  # Approximate focal length
    theta = np.radians(angle)
    
    # Rotation matrix for Y-axis
    rotation_matrix_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Map the original 2D points to 3D
    src_points = []
    for y in range(h):
        for x in range(w):
            src_points.append([x - w // 2, y - h // 2, f])  # Assume z = focal length (f)
    src_points = np.array(src_points)
    
    # Apply rotation
    rotated_points = src_points @ rotation_matrix_y.T
    
    # Project back to 2D
    projected_points = np.zeros((h * w, 2), dtype=np.float32)
    transformed_image = np.zeros_like(image)
    for i, point in enumerate(rotated_points):
        if point[2] > 0:  # Avoid division by zero
            x = int(point[0] / point[2] * f + w // 2)
            y = int(point[1] / point[2] * f + h // 2)
            if 0 <= x < w and 0 <= y < h:  # Clamp to image bounds
                original_x, original_y = divmod(i, w)
                if 0 <= original_x < w and 0 <= original_y < h:
                    transformed_image[int(y), int(x)] = image[original_y, original_x]
    
    return transformed_image


def augment_image(image):
    balek_image = flip_image(image, 1)
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # # Rotate
    # augmented_images.append(rotate_image(image, 90))
    # augmented_images.append(rotate_image(image, -90))

    # Scale
    augmented_images.append(scale_image(image, 2))
    augmented_images.append(scale_image(image, 1))
    augmented_images.append(scale_image(balek_image, 2))
    augmented_images.append(scale_image(balek_image, 1))
   

    # Translate
    augmented_images.append(translate_image(image, 20, 0))
    augmented_images.append(translate_image(image, -20, 0))
    augmented_images.append(translate_image(image, 0, 20))
    augmented_images.append(translate_image(image, 0, -20))
    augmented_images.append(translate_image(balek_image, 20, 0))
    augmented_images.append(translate_image(balek_image, -20, 0))
    augmented_images.append(translate_image(balek_image, 0, 20))
    augmented_images.append(translate_image(balek_image, 0, -20))

    # Flip
    augmented_images.append(flip_image(image, 1))  # Horizontal flip
    augmented_images.append(flip_image(balek_image, 1))  # Horizontal flip
    # augmented_images.append(flip_image(image, 0))  # Vertical flip

    # Brightness and Contrast
    augmented_images.append(adjust_brightness_contrast(image, 30, 30))
    augmented_images.append(adjust_brightness_contrast(image, 90, 90))
    augmented_images.append(adjust_brightness_contrast(image, -80, -80))
    augmented_images.append(adjust_brightness_contrast(image, -120, -120))
    augmented_images.append(adjust_brightness_contrast(balek_image, 30, 30))
    augmented_images.append(adjust_brightness_contrast(balek_image, 90, 90))
    augmented_images.append(adjust_brightness_contrast(balek_image, -80, -80))
    augmented_images.append(adjust_brightness_contrast(balek_image, -120, -120))


    # Gaussian Noise
    augmented_images.append(add_gaussian_noise(image, 0, 15))
    augmented_images.append(add_gaussian_noise(balek_image, 0, 15))

    # Crop
    h, w = image.shape[:2]
    augmented_images.append(crop_image(image, 20, 20, w-40, h-40))
    augmented_images.append(crop_image(balek_image, 20, 20, w-40, h-40))

    # Perspective Transform
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points[1] += (20, 0)
    dst_points[2] += (0, 20)
    augmented_images.append(perspective_transform(image, src_points, dst_points))
    augmented_images.append(perspective_transform(balek_image, src_points, dst_points))

    # Color Jitter
    augmented_images.append(color_jitter(image))
    augmented_images.append(color_jitter(balek_image))

    # # Black and White
    # augmented_images.append(convert_to_grayscale(image))

    # Perspective Rotate (Simulate Right/Left View)
    augmented_images.append(perspective_rotate_y(image, 30))  # Rotate to the right
    augmented_images.append(perspective_rotate_y(image, -30))  # Rotate to the left
    augmented_images.append(perspective_rotate_y(balek_image, 30))  # Rotate flipped to the right
    augmented_images.append(perspective_rotate_y(balek_image, -30))  # Rotate flipped to the left

    return augmented_images

known_directory = "known_faces"
output_directory = "../augmented_faces"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)
            
# Process each photo in the known directory
for folder in tqdm(os.listdir(known_directory), desc="Processing Folders"):
    input_folder_path = os.path.join(known_directory, folder)
    output_folder_path = os.path.join(output_directory, folder)
    
    # Ensure subfolder exists in augmented_faces
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Use tqdm for the file loop
    for filename in tqdm(os.listdir(input_folder_path), desc=f"Processing Images in {folder}", leave=False):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder_path, filename)
            image = cv2.imread(image_path)


            faces = face_detector(image)
            if len(faces) == 0:
               
                os.remove(image_path)
                continue

            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                face_frame = image[y:y + h, x:x + w]

    
            # Augment the image
            augmented_images = augment_image(face_frame)

            # Save augmented images in the corresponding folder
            for i, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
                cv2.imwrite(output_path, augmented_image)
