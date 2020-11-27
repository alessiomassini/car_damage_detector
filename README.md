# Car damage detector
We trained a `yolov4` model using `https://github.com/AlexeyAB/darknet.git` and built an effective tool to predict which parts of a car are damaged or whole. The parts that the detector recognizes are:
- front, rear and side body
- front, rear and side window
- wheels
- lights

The mAP of the detector is 80.45% in our valdation set.

The program has been tested on MacOS and Linux. For Windows users we recommend to use the Docker image that you can find at this link:

## Usage
To predict the images already included in our repo with the folder name `test_images`

1. Clone the repository
2. Inside the repository, launch 
    ```sh
    $ python3 setup.py
    ```
3. Go inside the darknet folder
    ```
    $ cd darknet
    ```
4. Launch the detector:
    ```
    $ python3 detector.py
    ```

## Predict your own images

The Python script `detector.py` has two changeable parameters:
- `imdir (default='test_images')`: string parameter that indicates the name of the folder where your own images to predict are located.
- `nms_tolerance (default=0.45)`:float parameter that indicates the maximum IoU tolerated between two predicted bounding boxes in order to keep both of them

Therefore, if you want to predict your own images, create a folder with your images and move it in car_damage_detector folder. Now just run
```
$ python3 detector.py -imdir='<name_of_your_folder>'
```
And the job is done!
