import numpy as np
import cv2
import torch
import lab1.lab1_cpu_gpu_test


def tensor_to_np(torch_tensor):
    np_arr = torch_tensor.detach().cpu().numpy()
    return np_arr


def bilinear_interpolation(image, new_size):
    # Получаем размеры исходного и нового изображений
    height, width = image.shape[:2]
    new_height, new_width = new_size

    # Вычисляем коэффициенты масштабирования по каждой оси
    scale_factor = float(width) / new_width
    tensor = lab1.lab1_cpu_gpu_test.mat_copy_gpu(image)
    # Инициализируем новое изображение заданного размера
    interpolation_image = torch.nn.functional.interpolate(input=tensor, scale_factor=2.0, mode='nearest')

    return interpolation_image


image = cv2.imread('test.png')
print(image)

# Указание нового размера (увеличение в 2 раза)
new_size = (image.shape[0] * 2, image.shape[1] * 2)
print(new_size)
# Билинейная интерполяция
interpolated_image = tensor_to_np(bilinear_interpolation(image, new_size))

# Сохранение исходного и интерполированного изображений
cv2.imwrite('Interpolated Image2.jpg', interpolated_image)
