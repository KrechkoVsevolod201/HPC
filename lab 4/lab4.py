import numpy as np
import cv2


def bilinear_interpolation_cpu(image, new_size):
    # Получаем размеры исходного и нового изображений
    height, width = image.shape[:2]
    new_height, new_width = new_size

    # Вычисляем коэффициенты масштабирования по каждой оси
    scale_x = float(width) / new_width
    scale_y = float(height) / new_height

    # Инициализируем новое изображение заданного размера
    interpolation_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Находим координаты соответствующие текущей позиции в новом изображении
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5

            # Находим координаты ближайших пикселей в исходном изображении
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, width - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, height - 1)

            # Находим весовые коэффициенты для билинейной интерполяции
            dx = src_x - x0
            dy = src_y - y0

            # Вычисляем значения пикселей в новом изображении с помощью билинейной интерполяции
            interpolation_image[y, x] = (1 - dx) * (1 - dy) * image[y0, x0] \
                                        + dx * (1 - dy) * image[y0, x1] \
                                        + (1 - dx) * dy * image[y1, x0] \
                                        + dx * dy * image[y1, x1]

    return interpolation_image


# Загрузка исходного изображения
image = cv2.imread('test.png')

# Указание нового размера (увеличение в 2 раза)
new_size = (image.shape[0] * 2, image.shape[1] * 2)
print(new_size)
# Билинейная интерполяция
interpolated_image = bilinear_interpolation_cpu(image, new_size)

# Сохранение исходного и интерполированного изображений
cv2.imwrite('Interpolated Image2.jpg', interpolated_image)