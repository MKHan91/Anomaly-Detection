from glob import glob

import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='make video')
parser.add_argument('--mode',       type=str,       default='VIDEO')
parser.add_argument('--image_dir',  type=str,       default='/home/onepredict/Myungkyu/BVMS_turbine/01_Training_Result/03_Output_image/BTT_AE_2019_11_25_15_49_53')

args = parser.parse_args()

def make_video(img_dir):
    images = [img for img in sorted(os.listdir(img_dir))]
    for image in sorted(images):
        image_paths = sorted(glob(os.path.join(img_dir, image, '*.png')))

        for idx, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path)
            height, width, _ = frame.shape

            video = cv2.VideoWriter(img_dir+'/Anomaly score variation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height), True)
            video.write(frame)

            print('converting!: {}'.format(idx + 1))
            video.release()

if __name__ == '__main__':
    if args.mode == 'VIDEO':
        make_video(args.image_dir)