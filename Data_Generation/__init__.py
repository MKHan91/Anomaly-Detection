import argparse

# BASE_PATH ='D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS deep learning development\\02_Training_Data'
BASE_PATH ='/home/onepredict/Myungkyu/LG_CNS/LG_CNS_data/By_datasetMK/20190411_normal_image'
# TEST_PATH = 'D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS deep learning development\\03_Test_Data'
TEST_PATH = '/home/onepredict/Myungkyu/BVMS_turbine/01_About_data/02_Test_Data'
# SPs_PATH = 'D:\\Onepredict_MK\\16 GuardiOne Turbine BVMS deep learning development\\Superpixel'

parser = argparse.ArgumentParser(description='init code of Myungkyu')
parser.add_argument('--mode',           type=str,   help='test or train data generation', default='Acoustics')
parser.add_argument('--data_mode',      type=str,   help='normal or fault or both',       default='both')

args = parser.parse_args()