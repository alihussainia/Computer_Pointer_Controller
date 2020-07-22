# Computer-Pointer-Controller
Control your mouse pointer with your eyes

## Description

This is the final project of Udacity's Intel Edge AI Nano Degree program in which all the students were tasked to implement a computer-pointer control program using Intel OpenVINO that utlizes [face detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html), [head pose estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html), [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html) and [gaze estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to estimate and control the  mouse cursor movements virtually. With the capability of running multiple models at once and successfully synchronizing the information flow coming from different models, make the program a good candidate to be implemented in real world case studies in order to solve the human centric computer-vision related business problems.

**Author**: Muhammad Ali

**Contact**: malirashid1994@gmail.com

**Profile**:

LinkedIn:- [LinkedIn](https://www.linkedin.com/in/alihussainia/)

**Date Of Submission**: 22th July, 2020.

## Project Set Up and Installation
- Activate the Virtual Environment `env` using:
    - `source env/bin/activate`
- Use below cmd to setup the environment variable by running:
    - `source /opt/intel/openvino/bin/setupvars.sh`
- Use the below cmd to install from the requirements.txt in the project root.
    - `pip3 install -r requirements.txt`
- Visit the [Demo](#demo_id) section to run the demo

## Prerequisites 

Primarily you need Intel's OpenVino toolkit installed on your machine.

**Required Packages**:
(mentioned in requirements.txt)
```
image==1.5.27
ipdb==0.12.3
ipython==7.10.2
numpy==1.17.4
Pillow==6.2.1
requests==2.22.0
virtualenv==16.7.9
pyautogui==0.9.50
opencv-python==4.2.0.34
```
## Directory Structure

The complete project directory structure is defined below:

```
|- resources                            # stores the demo files and images referenced in the README.md
|   |- demo.mp4
|
|- models                               # (not included) contains the Intermediate Representation of the models
|                                      
|- main.py                              # This file needs to run for testing/demo-ing the project
|- face_detection.py                    # Model Specific Wrapper Class files derived from provided model.py artifact
|- facial_landmarks_detection.py        # and then adapted as per the network to be used with.
|- gaze_estimation.py
|- head_pose_estimation.py
|- input_feeder.py                      # For capturing the input feed from either video or camera.
|- mouse_controller.py                  # Abstraction file for controlling the mouse pointer on screen.
|
|- README.md                            # For documentation
|- requirements.txt                     # Required packages list
```

### Models download

With the help of the OpenVino toolkit, we are required to download the 4 models used in this project.
The default values in the code suppose you execute these commands from the "model" directory.

The directory structure will be :

```
|- models                            
|   |- intel
|   |  |-face-detection-adas-binary-0001
|   |  | |-FP32-INTI
|   |  |
|   |  |-gaze-estimation-adas-0002                              
|   |  |  | |-FP16
|   |  |  | |-FP16-INTB
|   |  |  | |-FP32
|   |  |
|   |  |-head-pose-estimation-adas-0001  
|   |  |  | |-FP16
|   |  |  | |-FP16-INTB
|   |  |  | |-FP32
|   |  |   
|   |  |-landmarks-regression_retail_0009
|   |  |  | |-FP16
|   |  |  | |-FP16-INTB
|   |  |  | |-FP32-FP16
```

But you are of course free to use any compatible model, and specify these with the corresponding parameters.

#### Windows syntax example

This will download the models in all precisions available. You can add a ```--precisions``` argument if you only want a specific precision. These example are for the default installation of OpenVino 2020.2.117 using jupyter notebook i.e. Environmental Setup.ipynb present in the directory.

`!python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001`

`!python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001`

`!python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009`

`!python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002`
 

The total files size is about 41MB.


<h2 id="demo_id"> Demo </h2>

A basic usage should be as simple as this command :
```python main.py```

Without any parameters, the script will run in webcam mode (default) and will try to use models in the expected directories.

See below to specify other parameters.

## Documentation

With CLI arguments, you can specify other models or precisions (FP32 is used by default).

There are four arguments for specifying the 4 models used in this project:
- ```--face_detection_model```
- ```--gaze_estimation_model```
- ```--head_pose_estimation_model```
- ```--facial_landmarks_detection_model```

**Note:
You have to pass only the model's names i.e. without the .bin or .xml extensions.**

Another important argument is the ```--input_type```. It allows one of the three values to be passed at a time: 
- ```cam``` (default) 
- ```video``` 
- ```image```.

**Note:
The last 2 values specifically require another argument ```--input_file``` to be passed in order to identify the file as well as the path to the appropriate file.**

The ```--device``` argument (default=**CPU**) allows the program to use different hardware options to execute the inference such as GPU, MYRIAD, etc, as provided by the OpenVino framework.

The ```--extensions``` argument allows the program to make use of OpenVino extension.

The```--show_face``` argument (default=**True**) allows to disable the display of the detected face in a window.

Finally, the performance can be analyzed via using ```--perf_counts``` argument (default=**False**) that displays vital statistics on the terminal related to inferences performance for each model used.

## Benchmarks

With an i5-6500 CPU, I measured the following performances:
```
|- CPU & FP32 precisions : 
|  |- loading time = 686ms / Average total inferences time: 13ms
|
|- CPU & FP16 precisions : 
|  |- loading time = 685ms / Average total inferences time: 13ms
|
|- CPU & INT8 precisions : 
|  |- loading time = 850ms / Average total inferences time: 12ms
|
|- GPU & FP32 precisions : 
|  |- loading time = 51.0s / Average total inferences time: 25ms
|
|- GPU & FP16 precisions : 
|  |- loading time = 52.5s / Average total inferences time: 25ms
|
|- GPU & INT8 precisions : 
|  |- loading time = 54.5s / Average total inferences time: 26ms
|
|- CPU + MYRIAD on Gaze Estimation & FP16 : 
|  |- loading time = 2s / Average total inferences time: 17ms
|
|- CPU + MYRIAD on Facial Landmarks Detection & FP16 : 
|  |- loading time = 2s / Average total inferences time: 17ms
|
|- CPU + GPU on Face Detection & FP16 : 
|  |- loading time = 32s / Average total inferences time: 22ms
```
**Notes:** 
- The Face Detection model contains only FP32-INT1 precision.
- The "Average total inferences time" is for the sum of 4 models inferences.

## Results

Since the benchmarks observation clearly depicts that the Model Loading Time of CPUs is the least in all cases and that of GPUs are most. This is because GPUs are more computationally complex devices than the CPUs. However, CPUs, while being slow in terms of compute, can load different instruction sets faster. In case of MYRIAD, they are better than GPUs but not than the CPUs with respect to loading time. The CPUs offer a moderate performance with Intel Core i5 6th Gen being better.

It is advisable to use FP16 on Edge Devices requiring higher accuracy. FP32 is a poor fit on edge devices. In the same vector space, FP16 precision can help perform double the number of floating point operations. It, thus, offers a good balance between precisions.

## Edge Cases

With having multiple faces in the frame, the face detected first is used. Considering the probability of a change of a person whose face is detected, a constant target of using "the first face" is preferable. Moreover, a default probability threshold allows removing false inferences.