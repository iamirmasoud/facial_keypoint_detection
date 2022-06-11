# Facial Keypoint Detection using PyTorch on COCO Dataset

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses


Implementation
The complete computer vision pipeline consists of:

Detecting faces on the image with OpenCV Haar Cascades.
Detecting 68 facial keypoints with CNN with architecture based on this paper.

In this project, I build a facial keypoint detection system. The system consists of a face detector that uses Haar Cascades and a Convolutional Neural Network (CNN) that predict the facial keypoints in the detected faces. The facial keypoint detection system takes in any image with faces and predicts the location of 68 distinguishing keypoints on each face. The facial keypoints dataset used to train, validate and test the model consists of 5770 color images from the [ YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/). 


## Results

I got the best results when doing transfer learning on a [pretrained resnet18 torchvision model](https://pytorch.org/docs/stable/torchvision/models.html).

### Test images
Here are the predicted keypoints using the resnet model on the provided test images.

<img src="images/obamas_resnet.png" width="512">
<img src="images/beatles_resnet.png" width="512">
<img src="images/mona_lisa_resnet.png" width="255">

### Webcam app

In addition to testing the model on the test images, I developed a small webcam application that runs in the notebook ```Keypoint Webcam.ipynb```. The application detects a face from your webcam feed and uses the model to predict the keypoints and draws a simple mask onto the face. The app has decent performance on my face, and runs with around 4-5 FPS on my computer.

<img src="videos_and_gifs/face_mask_test.gif?" width="512"><br>

I also tried this on Paddy the cat, but the face detector, which uses Haar Cascades, is designed for human faces, and only detects the cats face periodically. But when the cat gets recognized, the predictions aren't too bad.

<img src="videos_and_gifs/cat_mask.gif?" width="512">

---


In this project, I'll build a facial keypoint detection system that takes in any image with faces, recognizes and detects faces, and predicts the location of 68 distinguishing keypoints on each face!

__Background:__

_Facial keypoints_ include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. 


## Getting the Files

### Data

All the data you'll need to train a neural network should be placed in the subdirectory `data`. To get the data, run the following commands in your terminal:

```
mkdir data

wget -P data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip

unzip -n data/train-test-data.zip -d data
```

### Model Download

You can use my pre-trained model for your own experimentation. To use it, [download](https://www.dropbox.com/s/peuk41xdy90z51o/keypoints_model.pt?raw=1) the model and placed in the subdirectory `saved_models`.

## Result

Here are some visualizations of the facial recognition, keypoints detection, CNN feature maps, and interesting sunglasses layover applications:

![Facial Keypoint Detection](images/result.png)


## Dependencies

Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python 3, PyTorch and its torchvision, OpenCV, Matplotlib, and tqdm.

## Preparing the environment
**Note**: I have developed this project on __Linux__. It can surely be run on Windows and Mac with some little changes.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/facial_keypoint_detection.git
cd facial_keypoint_detection
```

2. Create (and activate) a new environment, named `keypoint_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n keypoint_env python=3.7
	source activate keypoint_env
	```
	
	At this point your command line should look something like: `(keypoint_env) <User>:facial_keypoint_detection <user>$`. The `(keypoint_env)` indicates that your environment has been activated, and you can proceed with further package installations.

6. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, Matplotlib. You can install  dependencies using:
```
pip install -r requirements.txt
```

7. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd facial_keypoint_detection
```

8. Open the directory of notebooks, using the below command. You'll see all the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

9. Once you open any of the project notebooks, make sure you are in the correct `keypoint_env` environment by clicking `Kernel > Change Kernel > keypoint_env`.


## Dataset
### About MS COCO dataset
The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![Sample Coco Example](images/coco-examples.jpg)

You can read more about the dataset on the [website](http://cocodataset.org/#home), [research paper](https://arxiv.org/pdf/1405.0312.pdf), or Appendix section at the end of this page.

## Jupyter Notebooks
The project is structured as a series of Jupyter notebooks that should be run in sequential order:

### [0. Dataset Exploration notebook](0_Dataset_Exploration.ipynb) 

This notebook initializes the [COCO API](https://github.com/cocodataset/cocoapi) (the "pycocotools" library) used to access data from the MS COCO (Common Objects in Context) dataset, which is "commonly used to train and benchmark object detection, segmentation, and captioning algorithms."

### [1. Architecture notebook](1_Architecture.ipynb) 

This notebook uses the pycocotools, torchvision transforms, and NLTK to preprocess the images and the captions for network training. It also explores details of EncoderCNN, which is taken pretrained from [torchvision.models, the ResNet50 architecture](https://pytorch.org/docs/master/torchvision/models.html#id3). The implementations of the EncoderCNN and DecoderRNN are found in the [model.py](model.py) file.

The core architecture used to achieve this task follows an encoder-decoder architecture, where the encoder is a pretrained ResNet CNN on ImageNet, and the decoder is a basic one-layer LSTM.



## Results
Here are some predictions from the model.

### Some good results
![sample_171](images/sample_171.png?raw=true)<br/>
![sample_440](images/sample_440.png?raw=true)<br/>
![sample_457](images/sample_457.png?raw=true)<br/>
![sample_002](images/sample_002.png?raw=true)<br/>
![sample_029](images/sample_029.png?raw=true)<br/>
![sample_107](images/sample_107.png?raw=true)<br/>
![sample_202](images/sample_202.png?raw=true)


### Some not so good results

![sample_296](images/sample_296.png?raw=true)<br/>
![sample_008](images/sample_008.png?raw=true)<br/>
![sample_193](images/sample_193.png?raw=true)<br/>
![sample_034](images/sample_034.png?raw=true)<br/>
![sample_326](images/sample_326.png?raw=true)<br/>
![sample_366](images/sample_366.png?raw=true)<br/>
![sample_498](images/sample_498.png?raw=true)

## Future work
Steps for additional improvement would be exploring the hyperparameter and other architectures and also training with more epochs.

## Appendix: More about COCO dataset API
COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
- Please download, unzip, and place the images in: coco/images/
- Please download and place the annotations in: coco/annotations/

For substantially more details on the API please see [COCO Home Page](http://cocodataset.org/#home).

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
- For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
- For Python, run "make" under coco/PythonAPI
- For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/


Note: This project is a part of [Udacity Computer Vision Nanodegree Program](https://www.udacity.com/course/computer-vision-nanodegree--nd891).