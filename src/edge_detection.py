import numpy as np
import cv2


def gaussian_blur(image):
    """ Apply Gaussian blur to the input image and normalize the output. """
    # Define Gaussian kernel
    gaussian_kernel = np.array([[1 / 16, 2 / 16, 1 / 16],
                                [2 / 16, 4 / 16, 2 / 16],
                                [1 / 16, 2 / 16, 1 / 16]], dtype=np.float32)

    # Convolve the image with the Gaussian kernel
    blurred_image = convolve(image, gaussian_kernel)

    # Normalize to 0-255
    gaussian_image = np.clip(blurred_image, 0, 255).astype(np.uint8)

    return gaussian_image


def convolve(image, kernel):
    """ Apply convolution between the image and a kernel. """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    convolved_image = np.zeros((image_height, image_width), dtype=np.float32)

    # Pad the image to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    # Convolution operation
    for i in range(image_height):
        for j in range(image_width):
            # Apply the kernel
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved_image[i, j] = np.sum(region * kernel)

    return convolved_image


def sobel_edge_detection(image):
    """ Perform Sobel edge detection. """
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # Convolve the image with the Sobel kernels
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize pixels to 0-255
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude


def prewitt_edge_detection(image):
    """ Perform Prewitt edge detection. """
    # Define Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]])

    # Convolve the image with the Prewitt kernels
    grad_x = convolve(image, prewitt_x)
    grad_y = convolve(image, prewitt_y)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize pixels to 0-255
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_magnitude


def laplacian_edge_detection(image):
    """ Perform Laplacian edge detection. """
    # Define Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Convolve the image with the Laplacian kernel
    laplacian_image = convolve(image, laplacian_kernel)

    # Normalize to 0-255
    laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)

    return laplacian_image

def detect_edges(initial_image, kernel):
    """ Load an image, apply convolution with the given kernel, and return the result. """
    gray_scale_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    image = gaussian_blur(gray_scale_image)

    if kernel == 'sobel':
        edges = sobel_edge_detection(image)
    elif kernel == 'prewitt':
        edges = prewitt_edge_detection(image)
    elif kernel == 'laplacian':
        edges = laplacian_edge_detection(image)
    else:
        raise ValueError("Unsupported kernel type.")

    return edges
