# Atlas 200 DK Hand Pose Estimation

Code to execute the open pose model on the Atlas 200 DK board.

## Getting Started

Make sure you have access to a Atlas200DK, that's the first step. Look through this file to get setup with the rest

### Requirements
See the *requirements.txt* file or simply run:

    pip install -r requirements.txt

### File structure
.  
├── model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Contains the CaffeModel files**  
├── inputs
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Containes the input video examples** 
├── outputs   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Contains the resulting processed videos**  
├── protos
├── Results  
└── utils  


### While logged onto an Atlas, run the following commands to run the application

To run the als.mp4 file, simply run (using python3):

    python openposerun_optimized_delayshift.py
    
To run your's, or another example file, run:

    python openposerun_optimized_delayshift.py --input PATH/TO/FILE.mp4
    
There's no need to download, or setup an offline model file, the python code will automatically do it for you if it's not already there. For different
resolutions of videos, a new offline model will be created.


## Multi-threading Approach



## Performance

Depends largely on the input video, for my tests, it ranged from 15-24 FPS. I would encourage you to modify the `imHeight` variable to create a larger or smaller model. The accuracy is proportional to the size of the image, and the frame-rate is inversely proportional to the size of the image (video stream).
