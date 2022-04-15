# Atlas 200 DK Hand Pose Estimation

Code to execute the open pose model on the Atlas 200 DK board.

## Getting Started

Make sure you have access to a Atlas200DK, that's the first step. Look through this file to get setup with the rest

### Requirements
See the *requirements.txt* file or simply run:

    pip install -r requirements.txt

### File structure
.  
├── model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Contains the CaffeModel files**  
├── inputs
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Containes the input video examples**  
├── outputs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Contains the resulting processed videos**  
├── atlas_utils
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Utilities for running on the Atlas Board**  
├── optimization_strategies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Additional Multi-processing approaches**  
└── openposerun_optimized_delayshift.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Python file to run the application**  


### While logged onto an Atlas, run the following commands to run the application

To run the als.mp4 file, simply run (using python3):

    python openposerun_optimized_delayshift.py
    
To run your's, or another example file, run:

    python openposerun_optimized_delayshift.py --input PATH/TO/FILE.mp4
    
There's no need to download, or setup an offline model file, the python code will automatically do it for you if it's not already there. For different
resolutions of videos, a new offline model will be created.


## Multi-threading Approach

![method5](https://user-images.githubusercontent.com/33738542/163521767-ba8ca936-b61a-4490-a2f5-99067a50f0f7.png)

## Performance

Depends largely on the input video, for my tests, it ranged from 15-24 FPS. I would encourage you to modify the `imHeight` variable to create a larger or smaller model. The accuracy is proportional to the size of the image, and the frame-rate is inversely proportional to the size of the image (video stream).

## References

This project was adapted from Vikas Gupta's tutorial on Hand Keypoint Detection using Deep Learning and OpenCV. It was proposed by Huawei and completed as a part of graduation requirements for the Professional Masters of Computer Science degree at SFU.

```
@misc{gupta_2018, title={Hand keypoint detection using Deep Learning and opencv}, url={https://learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/}, journal={LearnOpenCV}, author={Gupta, Vikas}, year={2018}, month={10}} 
```
