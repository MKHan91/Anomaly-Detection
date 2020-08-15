from skimage.io import imread, imsave
from skimage.segmentation.slic_superpixels import slic
from skimage.segmentation import mark_boundaries
from skimage import color
from glob import glob
import matplotlib.pyplot as plt
import Data_Generation as DG
import os

BASE_PATH = DG.BASE_PATH
SPs_PATH = DG.SPs_PATH
mode = DG.args.mode
# COMPACTNESS = [20, 40, 80, 160]
COMPACTNESS = [40, 50, 60, 70, 80, 90, 100]
KIND = ['overlay', 'avg']

def superpixel_sample():
    image = imread(fname=BASE_PATH + '\\Training_Image\\00001.jpg')
    for kind in KIND:
        if kind == 'overlay':
            for idx, cp in enumerate(COMPACTNESS):
                "n_segments chooses the number of centers for kmeans," \
                "compactness trades off color-similarity and proximity. if compactness is large, space proximity is weighted." \
                "then, ther is more superpixel shape."
                image_segments = slic(image=image, n_segments=100, compactness=cp)
                superpixels = color.label2rgb(label=image_segments, image=image, kind=kind)

                if idx >= 1:
                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 2)
                    plt.ylabel('CP:{}'.format(cp))
                    plt.imshow(image)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 1)
                    plt.imshow(image_segments)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1))
                    plt.imshow(superpixels)
                else:
                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 2)
                    plt.title("raw_img")
                    plt.ylabel('CP:{}'.format(cp))
                    plt.imshow(image)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 1)
                    plt.title('img_seg')
                    plt.imshow(image_segments)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1))
                    plt.title('overlay')
                    plt.imshow(superpixels)

            # plt.show()
            plt.savefig(SPs_PATH + '\\SPs_overlay_v2.png', dpi=128)

        elif kind == 'avg':
            for idx, cp in enumerate(COMPACTNESS):
                image_segments = slic(image=image, n_segments=100, compactness=cp)
                superpixels = color.label2rgb(label=image_segments, image=image, kind=kind)

                if idx >= 1:
                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 2)
                    plt.ylabel('CP:{}'.format(cp))
                    plt.imshow(image)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 1)
                    plt.imshow(image_segments)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1))
                    plt.imshow(superpixels)
                else:
                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 2)
                    plt.title("raw_img")
                    plt.ylabel('CP:{}'.format(cp))
                    plt.imshow(image)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1) - 1)
                    plt.title('img_seg')
                    plt.imshow(image_segments)

                    plt.subplot(len(COMPACTNESS), 3, 3 * (idx + 1))
                    plt.title('avg')
                    plt.imshow(superpixels)
            # plt.show()
            plt.savefig(SPs_PATH + '\\SPs_avg_v2.png', dpi=128)

def SPs_generation():
    raw_img_list = glob(os.path.join(BASE_PATH, 'Training_Image', '*.jpg'))
    for index, raw_img in enumerate(raw_img_list):
        image = imread(fname=raw_img)
        image_segments = slic(image=image, n_segments=100, compactness=60)
        superpixels = color.label2rgb(label=image_segments, image=image, kind='avg')

        imsave(fname=os.path.join(BASE_PATH, 'Training_superpixel', '{:05d}.png'.format(index+1)),
               arr=superpixels)
        print("Processing: {}/{}".format(index+1, len(raw_img_list)))

if __name__ == '__main__':
    if mode == 'actual':
        SPs_generation()
    elif mode == 'sample':
        superpixel_sample()