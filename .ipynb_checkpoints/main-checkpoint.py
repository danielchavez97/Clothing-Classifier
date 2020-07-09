from PIL import Image, ImageFilter, ImageOps
import numpy as np


class ImagePipeline:
    def __init__(self, filename):
        self.filename = filename

    def load_image(self, filename):
        return Image.open(filename)

    def convert_to_png(self, image):
        image.save('image.png')
        return Image.open('image.png')

    #turns out 150 is the sweet spot for the threshold
    def crop(self, image, thresh):
        image_data = np.asarray(image) 
        image_data_bw = image_data.max(axis=2)
        col = np.where(image_data_bw.min(axis=0)<thresh)[0]
        row = np.where(image_data_bw.min(axis=1)<thresh)[0]
        crop = (min(row), max(row), min(col), max(col))
        image_new = image_data[crop[0]:crop[1]+1, crop[2]:crop[3]+1,:]
        new_image = Image.fromarray(image_new)
        return new_image

    def resize(self, image, iteration):
        x,y = image.size
        if iteration == 1:
            if x < y:
                return image.resize((int(x*(28/y)),28))
            else:
                return image.resize((28, int(y*(28/x))))
        else:
            return image.resize((28,28))

    def gaussian_operator(self,image):
        filt = ImageFilter.GaussianBlur(radius=1)
        return image.filter(filt)

    def invert(self, image):
        return ImageOps.invert(image)

    def grayscale(self, image):
        return image.convert('L')

    def as_vector(self, image):
        return np.asarray(image)

    def transform(self):
        image = self.load_image(self.filename)
        image = self.convert_to_png(image)
        image = self.crop(image, 244)
        image = self.resize(image, 1)
        image = self.gaussian_operator(image)
        image = self.resize(image, 2)
        image = self.invert(image)
        image = self.grayscale(image)
        data = self.as_vector(image)
        return data.flatten()
