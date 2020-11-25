import os
import shutil
#os.system('pip install requirements.txt')
from google_drive_downloader import GoogleDriveDownloader as gdd

os.chdir(os.getcwd())
os.system('git clone "https://github.com/alessiomassini/car_damage_detector.git"')
os.system('git clone "https://github.com/AlexeyAB/darknet.git"')


shutil.move(os.getcwd()+"/car_damage_detector/detector.py",os.getcwd()+"/darknet/detector.py")
shutil.move(os.getcwd()+"/car_damage_detector/obj.data",os.getcwd()+"/darknet/data/obj.data")
shutil.move(os.getcwd()+"/car_damage_detector/obj.names",os.getcwd()+"/darknet/data/obj.names")
shutil.move(os.getcwd()+"/car_damage_detector/yolov4-obj.cfg",os.getcwd()+"/darknet/cfg/yolov4-obj.cfg")
shutil.move(os.getcwd()+"/car_damage_detector/test_images",os.getcwd()+"/test_images")

#os.system('rm -r "car_damage_detector"')

os.chdir('./darknet')
gdd.download_file_from_google_drive(file_id='1ebQ1ffbaX4LMCFgVnjpfFnsbIDm8Q_IU',
                                    dest_path='./cardamage_final.weights',
                                    showsize=True)

os.system('make')