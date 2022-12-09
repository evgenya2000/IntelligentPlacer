import os
import cv2
import numpy as np
from imageio.v2 import imread, imsave
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from skimage.filters import gaussian, threshold_local, try_all_threshold, threshold_otsu, threshold_mean
from scipy.ndimage import binary_fill_holes
from rembg.bg import remove


# Функция для загрузки изображений
def im_upload(directory):
    images = []
    for file in os.listdir(directory):
        images.append(imread(os.path.join(directory, file)))
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
        image = imread(os.path.join(directory, file))
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
            imgs_contours[count] = imgs_contours[count][y + 17:y + h - 17, x + 17:x + w - 17]
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


# Удалим фон и получим маску для всех изображений
def fill_masks_figure(imgs):
    masks = []
    for img in imgs:
        image_without_back = np.array(rgba2rgb(remove(img), background=(0, 0, 0)))
        mask = []
        for p_str in image_without_back:
            t_str = []
            for pixel in p_str:
                t_str.append(pixel[0] != 0 and pixel[1] != 0 and pixel[2] != 0)
            mask.append(t_str)
        masks.append(
            binary_closing(binary_opening(np.array(mask), footprint=np.ones((15, 15))), footprint=np.ones((5, 5))))
    return masks


# Функция, которая сравнивает площади
def test_correct_pol(imgs_ob, imgs_pol):
    checked = []
    for i in range(len(imgs_ob)):
        if np.count_nonzero(imgs_ob[i] == True) >= np.count_nonzero(imgs_pol[i] == True):
            checked.append('NO')
        else:
            checked.append('YES')
    return checked


# Посмотрим на длину и ширину прямоугольника вокруг объектов и сравним её с высотой и шириной фигуры
def rectangle_coordinates(imgs_obs, imgs_pols):
    conts_obs = []
    conts_pols = []
    for img in imgs_obs:
        cont_obs = {'n_pix_up': [0, 0], 'n_pix_down': [0, 0], 'n_pix_left': [0, len(img[0]) - 1], 'n_pix_right': [0, 0]}
        if np.count_nonzero(img == True) == 0:
            conts_obs.append(True)
        else:
            # Ищем нижний крайний пиксель
            i = len(img) - 1
            while i > 0:
                if np.count_nonzero(img[i] == True) != 0:
                    cont_obs['n_pix_down'] = [i, np.where(img[i] == True)[0][0]]
                    break
                i = i - 1
            # Ищем верхний крайний пиксель
            i = 0
            while i < len(img):
                if np.count_nonzero(img[i] == True) != 0:
                    cont_obs['n_pix_up'] = [i, np.where(img[i] == True)[0][0]]
                    break
                i = i + 1
            # Ищем правый и левый пиксели
            for n_str in range(len(img)):
                if np.count_nonzero(img[n_str] == True) != 0:
                    temp = np.where(img[n_str] == True)[0]  # Индексы вхождения в строку
                    if cont_obs['n_pix_left'][1] > temp[0]:
                        cont_obs['n_pix_left'] = [n_str, temp[0]]
                    if cont_obs['n_pix_right'][1] < temp[-1]:
                        cont_obs['n_pix_right'] = [n_str, temp[-1]]
            conts_obs.append(cont_obs)

    for img in imgs_pols:
        cont_pol = {'n_pix_up': [0, 0], 'n_pix_down': [0, 0], 'n_pix_left': [0, len(img[0]) - 1], 'n_pix_right': [0, 0]}
        if np.count_nonzero(img == True) == 0:
            conts_pols.append(True)
        else:
            # Ищем верхний крайний пиксель
            i = len(img) - 1
            while i > 0:
                if np.count_nonzero(img[i] == True) != 0:
                    cont_pol['n_pix_down'] = [i, np.where(img[i] == True)[0][0]]
                    break
                i = i - 1
            # Ищем нижний крайний пиксель
            i = 0
            while i < len(img):
                if np.count_nonzero(img[i] == True) != 0:
                    cont_pol['n_pix_up'] = [i, np.where(img[i] == True)[0][0]]
                    break
                i = i + 1
            # Ищем правый и левый пиксели
            for n_str in range(len(img)):
                if np.count_nonzero(img[n_str] == True) != 0:
                    temp = np.where(img[n_str] == True)[0]  # Индексы вхождения в строку
                    if cont_pol['n_pix_left'][1] > temp[0]:
                        cont_pol['n_pix_left'] = [n_str, temp[0]]
                    if cont_pol['n_pix_right'][1] < temp[-1]:
                        cont_pol['n_pix_right'] = [n_str, temp[-1]]

            conts_pols.append(cont_pol)

    return conts_obs, conts_pols


# Сравним высоту и ширину, чтобы исключить случаи, когда точно не поместится
def check_parameter_height_width(conts_obs, conts_pols):
    checked = []
    for cont_obs, cont_pol in zip(conts_obs, conts_pols):
        if cont_obs != True:
            h_obs = cont_obs['n_pix_down'][0] - cont_obs['n_pix_up'][0]
            w_obs = cont_obs['n_pix_right'][1] - cont_obs['n_pix_left'][1]
            h_pol = cont_pol['n_pix_down'][0] - cont_pol['n_pix_up'][0]
            w_pol = cont_pol['n_pix_right'][1] - cont_pol['n_pix_left'][1]
            if h_obs > h_pol or w_obs > w_pol:
                checked.append('NO')
            else:
                checked.append('YES')
        else:
            checked.append(True)
    return checked


# Находит вершины области на листе, в зависимости от сравниваемой крайней точки объектов
def abcd(flag, cont_obs, X, Y):
    if flag == 'up':
        A = {'x': X - (cont_obs['n_pix_up'][1] - cont_obs['n_pix_left'][1]),
             'y': Y}
        B = {'x': X + (cont_obs['n_pix_right'][1] - cont_obs['n_pix_up'][1]),
             'y': Y}
        C = {'x': A['x'],
             'y': A['y'] + (cont_obs['n_pix_down'][0] - cont_obs['n_pix_up'][0])}
        D = {'x': B['x'],
             'y': C['y']}
    elif flag == 'down':
        C = {'x': X - (cont_obs['n_pix_down'][1] - cont_obs['n_pix_left'][1]),
             'y': Y}
        D = {'x': X + (cont_obs['n_pix_right'][1] - cont_obs['n_pix_down'][1]),
             'y': Y}
        A = {'x': C['x'],
             'y': C['y'] - (cont_obs['n_pix_down'][0] - cont_obs['n_pix_up'][0])}
        B = {'x': D['x'],
             'y': A['y']}
    elif flag == 'left':
        C = {'x': X,
             'y': Y + (cont_obs['n_pix_down'][0] - cont_obs['n_pix_left'][0])}
        A = {'x': X,
             'y': Y - (cont_obs['n_pix_left'][0] - cont_obs['n_pix_up'][0])}
        B = {'x': A['x'] + (cont_obs['n_pix_right'][1] - cont_obs['n_pix_left'][1]),
             'y': A['y']}
        D = {'x': B['x'],
             'y': C['y']}
    elif flag == 'right':
        B = {'x': X,
             'y': Y - (cont_obs['n_pix_right'][0] - cont_obs['n_pix_up'][0])}
        D = {'x': X,
             'y': Y + (cont_obs['n_pix_down'][0] - cont_obs['n_pix_right'][0])}
        A = {'x': B['x'] - (cont_obs['n_pix_right'][1] - cont_obs['n_pix_left'][1]),
             'y': B['y']}
        C = {'x': A['x'],
             'y': D['y']}
    return A, B, C, D


def countor_pol_array(cont_pol, im_pol):
    con = []
    for i in range(cont_pol['n_pix_up'][0], cont_pol['n_pix_down'][0] + 1):
        for j in range(cont_pol['n_pix_right'][1] + 1):
            if im_pol[i][j] == True:
                if (i == 0 or i == len(im_pol) - 1):
                    con.append([i, j])
                elif (j == 0 or j == len(im_pol[i]) - 1):
                    con.append([i, j])
                elif (im_pol[i - 1][j] == False or im_pol[i + 1][j] == False or \
                      im_pol[i][j - 1] == False or im_pol[i][j + 1] == False):
                    con.append([i, j])
    return con


def mask_matching(cont_obs, cont_pol, im_obs, im_pol):
    result = ['NO']
    # Создадим массив пикселей обрамляющих фигуру на листе
    countor_polynom = countor_pol_array(cont_pol, im_pol)
    for i in range(0, len(countor_polynom), 15):
        # Сопостовляемая точка на листе из массива контура фигуры
        X = countor_polynom[i][1]  # cont_pol['n_pix_up'][1]
        Y = countor_polynom[i][0]  # cont_pol['n_pix_up'][0]
        for extrem in ['up', 'down', 'left', 'right']:
            # Пройдемся по этим пикселям 4 раза, используя 4 крайние точки объектов
            # Найдём точки сопоставляемой области листка
            A, B, C, D = abcd(extrem, cont_obs, X, Y)

            # Проверим, что они не выходят за рамки изображения с листком
            if A['x'] < 0 or A['y'] < 0 or B['x'] < 0 or B['y'] < 0 or C['x'] < 0 or C['y'] < 0 or D['x'] < 0 or D[
                'y'] < 0 or \
                    A['x'] >= len(im_pol[0]) or A['y'] >= len(im_pol) or \
                    B['x'] >= len(im_pol[0]) or B['y'] >= len(im_pol) or \
                    C['x'] >= len(im_pol[0]) or C['y'] >= len(im_pol) or \
                    D['x'] >= len(im_pol[0]) or D['y'] >= len(im_pol):
                temp_res = 'NO'
            else:
                # Найдём вершины прямоугольника с объектами
                a = {'x': cont_obs['n_pix_left'][1],
                     'y': cont_obs['n_pix_up'][0]}
                b = {'x': cont_obs['n_pix_right'][1],
                     'y': cont_obs['n_pix_up'][0]}
                c = {'x': a['x'],
                     'y': cont_obs['n_pix_down'][0]}
                d = {'x': b['x'],
                     'y': c['y']}
                # Вырежем его
                res_obs = []
                for i in range(a['y'], c['y'] + 1):
                    k = []
                    for j in range(a['x'], b['x'] + 1):
                        k.append(im_obs[i][j])
                    res_obs.append(k)

                res_i = []
                # Вырежем сопостовляемую область
                for i in range(A['y'], C['y'] + 1):
                    s = []
                    for j in range(A['x'], B['x'] + 1):
                        s.append(im_pol[i][j])
                    res_i.append(s)

                result[0] = 'YES'
                # Сопоставим 2 изображения
                for i in range(len(res_i)):
                    for j in range(len(res_i[i])):
                        if res_i[i][j] == False and res_obs[i][j] == True:
                            result[0] = 'NO'
                            break
                    if result[0] == 'NO':
                        break
            if result[0] == 'YES':
                break
        if result[0] == 'YES':
            result.append(np.logical_xor(res_i, res_obs))
            break

    return result
