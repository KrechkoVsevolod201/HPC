import numpy as np
import numba
import cv2
import lab1.lab1_cpu_gpu_test

@numba.njit
def bilinear_interpolation_gpu(image, output_shape, timer_on=True):
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = output_shape[:2]

    # Compute scale factors
    sh = src_h / dst_h
    sw = src_w / dst_w

    # Create output image
    output_image = np.zeros(output_shape, dtype=np.float32)

    for i in range(dst_h):
        for j in range(dst_w):
            # Calculate the coordinate in the source image
            y = i * sh
            x = j * sw

            # Compute the integer part of the coordinates
            y_int = int(y)
            x_int = int(x)

            # Compute the fractional part of the coordinates
            y_frac = y - y_int
            x_frac = x - x_int

            y_int_next = min(y_int + 1, src_h - 1)
            x_int_next = min(x_int + 1, src_w - 1)

            # Perform bilinear interpolation
            output_image[i, j] = (
                (1 - x_frac) * (1 - y_frac) * image[y_int, x_int] +
                (1 - x_frac) * y_frac * image[y_int_next, x_int] +
                x_frac * (1 - y_frac) * image[y_int, x_int_next] +
                x_frac * y_frac * image[y_int_next, x_int_next]
            )
    flag = 0

    '''
    for y in range(dst_h):
        print(y)
        if (flag <= int(1 / sh)):
            output_image[y, 0] = image[int(y * sh -1), 0]
            output_image[y, 1] = image[int(y * sh-1), 1]
            #print(image[1, int(y * scale_x)])
            flag = flag + 1
            if flag == int(1 / sh):
                flag = 0

    for x in range(dst_w):
        print(x)
        if (flag <= int(1 / sh)):
            output_image[0, x] = image[0, int(x * sh -1)]
            output_image[1, x] = image[1, int(x * sh-1)]
            #print(image[1, int(y * scale_x)])
            flag = flag + 1
            if flag == int(1 / sh):
                flag = 0
    '''
    return output_image


# Загрузка изображения
image = cv2.imread('smol.png', cv2.IMREAD_GRAYSCALE)

# Указание нового размера (увеличение в 2 раза)
new_size = (image.shape[0] * 2, image.shape[1] * 2)
print(new_size)

# Изменение размера изображения
resized_image = cv2.resize(image, new_size)

# Выполнение билинейной интерполяции
output_image = bilinear_interpolation_gpu(resized_image, new_size)

# Сохранение исходного и интерполированного изображений
cv2.imwrite('Interpolated Image2.jpg', output_image)
