import torch

import PIL
from PIL import Image, ImageDraw

import random

def ShearX(img: Image, magnitude: float) -> Image:
    print(img)
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), #random.choice([-1, 1]) == -1과 1중 랜덤하게 선택
        PIL.Image.BICUBIC
    )

def TranslateX(img: Image, magnitude: float) -> Image:
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0)
    )

def Rotate(img: Image, magnitude: float) -> Image:
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(
        rot, Image.new("RGBA", rot.size), rot).convert(img.mode)

def Contrast(img: Image, magnitude: float) -> Image:
    return PIL.ImageEnhance.Contrast(img).enhance(1+magnitude*random.choice([-1,1]))

def Invert(img: Image, magnitude: float) -> Image:
    return PIL.ImageOps.invert(img)

def Equalize(img: Image, magnitude: float) -> Image:
    return PIL.ImageOps.equalize(img)

def AutoContrast(img: Image, magnitude: float) -> Image:
    return PIL.ImageOps.autocontrast(img)

def Solarize(img: Image, magnitude: float) -> Image:
    return PIL.ImageOps.solarize(img, magnitude)

def Posterize(img: Image, magnitude: float) -> Image:
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)

def Color(img: Image, magnitude: float) -> Image:
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1,1]))

def Brightness(img: Image, magnitude: float) -> Image:
    return PIL.ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1,1]))

def Cutout(img: Image, magnitude: float) -> Image:
    if magnitude == 0.0:
        return img
    w, h = img.size
    xy = [(random.randint(0,w), random.randint(0,h)), (random.randint(0,w),random.randint(0,h))]
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=(0,0,0))
    return img


