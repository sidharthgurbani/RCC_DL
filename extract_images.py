import pydicom as dicom
from RunSegTest import run
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import os

def readImage():
    image_path = "../Dataset/Outputfiles/RCCPV035/RCCPV035_seg.jpg"

    # img = Image.open(image_path)
    # print(img.format)
    # img.show()

    img = mpimg.imread(image_path)
    print(img.shape)
    plt.imshow(img)
    plt.show()


def having_rt_struct_folder(dataset_dir):
    x = [dir.name for dir in os.scandir(dataset_dir) if dir.is_dir()
         for dir2 in os.scandir(dir) if dir2.is_dir() for dir3 in os.scandir(dir2) if dir3.is_dir()
         for folders in os.scandir(dir3) if folders.is_dir() and folders.name == 'RTSTRUCT']

    return x


def get_dcm_input_files(dataset_dir):
    x = [[dir.name, file_dir, file] for dir in os.scandir(dataset_dir) if dir.is_dir()
         for dir2 in os.scandir(dir) if dir2.is_dir() for dir3 in os.scandir(dir2) if dir3.is_dir()
         for folders in os.scandir(dir3) if folders.is_dir() and folders.name == 'RTSTRUCT'
         for file_dir in os.scandir(folders) if file_dir.is_dir() and 'UWHC_LDD157' in file_dir.name
         # file_dir.name == '200 -- HM_LESIONS BY UWHC_LDD157'
         for file in os.scandir(file_dir) if file.is_file()]

    return x


def generate_images_from_dcm(dataset_dir):
    stl_file_list = open('stl_files.txt', 'a')
    image_file_list = open('image_files.txt', 'a')
    dcm_input_files = get_dcm_input_files(dataset_dir=dataset_dir)

    ignore_list = ['RCCPV006', 'RCCPV030', 'RCCPV065', 'RCCPV031', 'RCCPV012', 'RCCPV112', 'RCCPV032', 'RCCPV068',
                   'RCCPV057', 'RCCPV074', 'RCCPV029', 'RCCPV011', 'RCCPV108']

    # df = pd.DataFrame(ignore_list)
    # df.to_excel("dcm_read__error.xlsx")

    for name, file_dir, input_file in dcm_input_files:
        if name in ignore_list:
            continue
        input_file_path = input_file.path
        print(file_dir.path)
        ds = dicom.read_file(input_file_path)
        print(ds)
        break
        # file_name = os.path.splitext(input_file.name)[0]
        # output_file = file_dir.path + '/seg_' + file_name + '_'
        # contour_sequence_name = ds.StructureSetROISequence[0].ROIName
        # output_file += contour_sequence_name + '.stl'
        # image_file = file_dir.path + '/seg_' + file_name + '_' + contour_sequence_name + '.jpg'
        # print(output_file)
        # run(input_file, output_file, contour_sequence_name)
        # stl_file_list.write(output_file + '\n')
        # image_file_list.write(image_file + '\n')

    return