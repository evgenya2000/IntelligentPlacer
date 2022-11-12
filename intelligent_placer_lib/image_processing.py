import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

#from cv2 import IMREAD_GRAYSCALE, cvtColor, COLOR_BGR2GRAY, threshold, findContours
from imageio.v2 import imread, imsave
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from skimage.filters import sobel, gaussian, threshold_local, try_all_threshold, threshold_otsu, threshold_minimum, threshold_mean
from scipy.ndimage import binary_fill_holes


# Функция для загрузки изображений
def im_upload(directory):
    images = []
    for file in os.listdir(directory):
        images.append(imread(os.path.join(directory,file)))
    return images

# Функция сжатия
def im_comp(images):
    scale_percent = 87  # степень сжатия
    icomp = []
    for image in images:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_size = (width, height)
        icomp.append(cv2.resize(image, new_size))
    return icomp


# Функция перевода в черно-белый формат
def set_grayscale(images):
    grayscale_images = []

    for img in images:
        # gray_image = rgb2gray(img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_images.append(gray_image)

    return grayscale_images

# Бинаризация
def binary_otsu(images):
    bin_images = []

    for img in images:
        thresh_otsu = threshold_otsu(img)
        bin_images.append(img <= thresh_otsu)

    return bin_images

# Морфологические операции
def morpholog(images):
    bin_open_images = []
    for img in images:
        bin_open_images.append(binary_closing(img, footprint=np.ones((5, 5))))

    return bin_open_images

# Загрузка и сжатие изображений
def loading_and_preprocessing(directory):
    images = []
    scale_percent = 87
    for file in os.listdir(directory):
        image = imread(os.path.join(directory,file))
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_size = (width, height)
        icomp = cv2.resize(image, new_size)
        images.append(icomp)
        
    return images

# Разделение изображения на массив пикселей
def get_images_arrs(images):
    arrs_images = []
    for image in images:
        arr_im = np.array(image)
        arrs_images.append(arr_im)
    return arrs_images

# Получение контуров
def get_contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=cv2.contourArea, reverse=True)

    return contours

# Получение контура листа
def get_contours_list(imgs):
    imgs_contours = get_images_arrs(imgs)

    for count, img in enumerate(imgs):
        contours = get_contours(img)
        if len(contours[0]) > 0:
            contour = contours[0]
            epsilon = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.07 * epsilon, True)

            cv2.drawContours(imgs_contours[count], [approx], -1, (255, 0, 0), 30)

    return imgs_contours

# Обрезка листа
def get_list(imgs):
    imgs_contours = get_images_arrs(imgs)

    for count, img in enumerate(imgs):
        contours = get_contours(img)
        if len(contours[0]) > 0:
            contour = contours[0]
            epsilon = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.07 * epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            imgs_contours[count] = imgs_contours[count][y + 5:y + h - 5, x + 5:x + w - 5]

    return imgs_contours


# Обрезка части с объектами
def get_objects(imgs):
    imgs_contours = get_images_arrs(imgs)

    for count, img in enumerate(imgs):
        contours = get_contours(img)
        if len(contours[0]) > 0:
            contour = contours[0]
            epsilon = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.003 * epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            imgs_contours[count] = imgs_contours[count][0:imgs_contours[count].shape[0],
                                   x + w:imgs_contours[count].shape[1]]

    return imgs_contours

# Маска обрезанного листа
def get_mask_list(imgs):
    g_imgs = set_grayscale(imgs)
    b_imgs = binary_otsu(g_imgs)
    mask_pol = morpholog(b_imgs)
#    get_opencv_images(ec_imgs)
    return mask_pol

# Получение самого большого компонета
def get_largest_component(mask):
    labels = sk_measure_label(mask)
    props = regionprops(labels)
    areas = [prop.area for prop in props]
    largest_comp_id = np.array(areas).argmax()
    return labels == (largest_comp_id + 1)

# Закраска фигуры
def fill_mask_figure(masks):
    fill_masks = []
    for mask in masks:
        tmp = get_largest_component(mask)
        fill_masks.append(binary_fill_holes(tmp))
    return fill_masks