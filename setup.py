import os
import shutil
os.system('pip install -r requirements.txt')
from google_drive_downloader import GoogleDriveDownloader as gdd

os.chdir(os.getcwd())
os.system('git clone "https://github.com/AlexeyAB/darknet.git"')

shutil.move(os.getcwd()+"/detector.py",os.getcwd()+"/darknet/detector.py")
shutil.move(os.getcwd()+"/obj.data",os.getcwd()+"/darknet/data/obj.data")
shutil.move(os.getcwd()+"/obj.names",os.getcwd()+"/darknet/data/obj.names")
shutil.move(os.getcwd()+"/yolov4-obj.cfg",os.getcwd()+"/darknet/cfg/yolov4-obj.cfg")

os.chdir('./darknet')
gdd.download_file_from_google_drive(file_id='1ebQ1ffbaX4LMCFgVnjpfFnsbIDm8Q_IU',
                                    dest_path='./cardamage_final.weights',
                                    showsize=True)

os.system("sed -i -- 's/AVX=0/AVX=1/g' Makefile")
os.system("sed -i -- 's/OPENMP=0/OPENMP=1/g' Makefile")
os.system('make')