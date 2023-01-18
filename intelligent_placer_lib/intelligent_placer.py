import sys
sys.path.append('./intelligent_placer_lib')
from image_processing import *

def check_image(path):
    result = []
    images = []
    scale_percent = 87

    # Загрузим изображение и сожмём его
    file = path.split('/')[-1]
    result.append(file)
    directory = path[0:len(path) - len(file) - 1]
    images.append(imread(os.path.join(directory, file)))
    width = int(images[0].shape[1] * scale_percent / 100)
    height = int(images[0].shape[0] * scale_percent / 100)
    new_size = (width, height)
    images[0] = cv2.resize(images[0], new_size)

    # Получим изображение листа с фигурой и изображение объектов
    images_list = get_list(images)
    images_objects = get_objects(images)

    # Получим маску фигуры
    images_pol = get_mask_list(images_list)
    fill_ims_pol = fill_mask_figure(images_pol)

    # Получим маску объектов
    images_without_back = fill_masks_figure(images_objects)

    # Сравним площади фигуры и объектов
    checked_correct_pol = test_correct_pol(images_without_back, fill_ims_pol)
    if checked_correct_pol[0] == 'NO':
        result.append('NO')
        return result
    else:
        # Получим вершины фигуры и прямоугольника, обрамляющего объекты
        contours_objects, contours_pols = rectangle_coordinates(images_without_back, fill_ims_pol)
        # Проверим, что объекты могут влезть в фигуру
        checked_param_countours = check_parameter_height_width(contours_objects, contours_pols)
        if checked_param_countours[0] == True:  # Вариант, когда нет объектов
            result.append('YES')
            result.append(fill_ims_pol[0])
            return result
        elif checked_param_countours[0] == 'NO':
            result.append('NO')
            return result
        else:
            result = [result[0]] + mask_matching(contours_objects[0], contours_pols[0], images_without_back[0],
                                                 fill_ims_pol[0])
    return result

