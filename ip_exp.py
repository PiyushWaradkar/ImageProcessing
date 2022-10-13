from tkinter import Label
import numpy as np
from numba import jit
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from math import sqrt, atan2, pi

@jit
def grayscale(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            r,g,b=img[i,j]
            sum=0.299*r+0.58*g+0.114*b
            img[i,j]=sum
    return img

#@jit for some reason cpu is much better then gpu in execution time
def negative_img(img):
    for i in range(len(img)):
        img[i]=255-img[i]
    return img

@jit
def gray_level_thresholding(img,A,B):
    img=grayscale(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if A <= img[i,j][0] <= B: 
                img[i,j]= 255
            else:
                img[i,j] = 0
    return img

@jit
def gray_level_slicing(img,A,B):
    img=grayscale(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if A <= img[i,j][0] <= B: 
                img[i,j]= 255

    return img

def bit_plane_splicing(img):
    img=grayscale(img)
    temp_img=img
    list=[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            dec=img[i,j][0]
            b=bin(dec)[2:].zfill(8)
            list.append(b)
            list.append(b)
            list.append(b)

    temp_img=np.reshape(list,img.shape)
    eight_img,seven_img,six_img,five_img,four_img,three_img,two_img,one_img=[],[],[],[],[],[],[],[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            b=temp_img[i,j][0]
            for _ in range(3):
                eight_img.append(int(b[0])*255)
                seven_img.append(int(b[1])*255)
                six_img.append(int(b[2])*255)
                five_img.append(int(b[3])*255)
                four_img.append(int(b[4])*255)
                three_img.append(int(b[5])*255)
                two_img.append(int(b[6])*255)
                one_img.append(int(b[7])*255)

    eight_img=np.reshape(eight_img,img.shape)
    seven_img=np.reshape(seven_img,img.shape)
    six_img=np.reshape(six_img,img.shape)
    five_img=np.reshape(five_img,img.shape)
    four_img=np.reshape(four_img,img.shape)
    three_img=np.reshape(three_img,img.shape)
    two_img=np.reshape(two_img,img.shape)
    one_img=np.reshape(one_img,img.shape)
        
    return [eight_img,seven_img,six_img,five_img,four_img,three_img,two_img,one_img]


def contrast_streaching(img,s1=0,s2=255):
    temp_img=img
    rmin=np.min(img)
    rmax=np.max(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            r=temp_img[i,j]
            r =s1 + (s2-s1)*(r-rmin)/(rmax-rmin)
            temp_img[i,j]=r

    return temp_img


def  hist(img):
    img = Image.fromarray(img)
    img2 = ImageOps.equalize(img, mask=None)
    histogram1 = img.histogram()
    histogram2 = img2.histogram()
    
    plt.plot(histogram1,label='orignal')
    plt.plot(histogram2,label='equalized')
    plt.legend()
    plt.show()
    
    return(img2)

def box_filter(img):
    img = img.filter(ImageFilter.Kernel((3, 3), (1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9), 1, 0))
    return img

def median_filter(img):
    img = img.filter(ImageFilter.MedianFilter)
    return img

def weighted_filter(img):
    img = img.filter(ImageFilter.SMOOTH)
    return img


def Sobel_filter(img):
    temp_img=img
    input_pixels = temp_img.load()
    intensity = [[sum(input_pixels[x, y]) / 3 for y in range(temp_img.height)]
                for x in range(temp_img.width)]
    kernelx = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
    kernely = [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]

    
    output_image = Image.new("RGB", temp_img.size)
    draw = ImageDraw.Draw(output_image)

    for x in range(1, temp_img.width - 1):
        for y in range(1, temp_img.height - 1):
            magx, magy = 0, 0
            for a in range(3):
                for b in range(3):
                    xn = x + a - 1
                    yn = y + b - 1
                    magx += intensity[xn][yn] * kernelx[a][b]
                    magy += intensity[xn][yn] * kernely[a][b]

            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))
    return output_image

def Prewitt_filter(img):
    temp_img=img
    input_pixels = temp_img.load()
    intensity = [[sum(input_pixels[x, y]) / 3 for y in range(temp_img.height)]
                for x in range(temp_img.width)]
    kernelx = [[-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]]
    kernely = [[-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]]

    
    output_image = Image.new("RGB", temp_img.size)
    draw = ImageDraw.Draw(output_image)

    for x in range(1, temp_img.width - 1):
        for y in range(1, temp_img.height - 1):
            magx, magy = 0, 0
            for a in range(3):
                for b in range(3):
                    xn = x + a - 1
                    yn = y + b - 1
                    magx += intensity[xn][yn] * kernelx[a][b]
                    magy += intensity[xn][yn] * kernely[a][b]

            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))
    return output_image

def Robert_filter(img):
    temp_img=img
    input_pixels = temp_img.load()
    intensity = [[sum(input_pixels[x, y]) / 3 for y in range(temp_img.height)]
                for x in range(temp_img.width)]
    kernelx = [[0, 0, 0],
            [0, -1, 0],
            [0, 0, 1]]
    kernely = [[0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]

    
    output_image = Image.new("RGB", temp_img.size)
    draw = ImageDraw.Draw(output_image)

    for x in range(1, temp_img.width - 1):
        for y in range(1, temp_img.height - 1):
            magx, magy = 0, 0
            for a in range(3):
                for b in range(3):
                    xn = x + a - 1
                    yn = y + b - 1
                    magx += intensity[xn][yn] * kernelx[a][b]
                    magy += intensity[xn][yn] * kernely[a][b]

            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))
    return output_image

def laplacian_sharpening(img):
    img = img.filter(ImageFilter.Kernel(
        (3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0))
    return img


def canny_main(img):
    output_image = Image.new("RGB", img.size)
    draw = ImageDraw.Draw(output_image)
    for x, y in canny_edge_detector(img):
        draw.point((x, y), (255, 255, 255))
    return output_image

def canny_edge_detector(img):
    input_pixels = img.load()
    width = img.width
    height = img.height

    grayscaled = compute_grayscale(input_pixels, width, height)

    blurred = compute_blur(grayscaled, width, height)

    gradient, direction = compute_gradient(blurred, width, height)

    filter_out_non_maximum(gradient, direction, width, height)

    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale

@jit
def compute_blur(input_pixels, width, height):

    def clip(x, l, u): return l if x < l else u if x > u else x

    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    offset = len(kernel) // 2

    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred

@jit
def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction

@jit
def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x,
                                                 y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0

@jit
def filter_strong_edges(gradient, width, height, low, high):
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep
        return list(keep)
