## Introduction
 As rookie parents to an 8 month old baby girl, we wanted to build an object recognition app to help her to learn more about this new world through the eye of the camera using YOLO, a cool algorithm we learned on our parental leave. BUT, there's always a BUT, Google beat us to it! (not surprising :stuck_out_tongue_closed_eyes: ) Oh, I mean, Google saved us lots of time and we can just build upon their work. 


## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

### Installation for Ubuntu (14.04)

#### Prerequisites for tensorfow with GPU support (with a friendly WARNING!!!) 
- You need to install NVIDIA CUDA Toolkit (>= 7) and for Ubuntu 14.04, 8.0 is the highest supported version.
- You need to install cuDNN v6.0
 
```
$ sudo apt-get install aptitude 
$ sudo aptitude install cuda (Trust me, it's the easiest way to install CUDA) 

$ export CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
$ wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
$ tar zxvf ${CUDNN_TAR_FILE}
$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
$ sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*
```

##### set environment variables in /etc/environment

```
PATH=/usr/local/cuda-8.0/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```

- And you have to reboot your Ubuntu to pick up the Nvidia library change :scream:, myself can't believe that either!
- Now you can install tensorflow

#### Install tensorflow 
- Installing tensroflow CPU version was the only solution for my Lenovo W541 + Nvidia graphics card because installing the libraries of CUDA and co caused my screen to freeze almost every minute. 
So I strongly recommend you installing tensorflow CPU version to get rid of the Nvidia dependencies, unless you need to do heavy training. 
- You can install `tensorflow` (CPU only) or `tensorflow-gpu` following the instructions below

```
$ sudo apt-get install python-pip3 python-dev
$ sudo pip3 install Cython
$ sudo pip3 install opencv-python
$ virtualenv --system-site-packages  (this works for python 3.n as well, if you run the other command recommended by the tensorflow document, you may hit a known issue caused by python package conflict) 
$ source ~/tensorflow/bin/activate
(tensorflow)$ easy_install -U pip
(tensorflow)$ pip3 install --upgrade tensorflow 
or
(tensorflow)$ pip3 install --upgrade tensorflow-gpu

```

- Now you're ready to run this project in the tensorflow virtual environment 


### Getting started

Always make sure to download the up-to-date weight files and the cfg files from from [Darknet](https://pjreddie.com/darknet/yolo/) because both of the formats may change at the same time.

- Download the YOLO V2 `yolo.weights` and the model file `yolo.cfg' here. 
```
wget http://pjreddie.com/media/files/yolo.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg
```

- Download the tiny YOLO V2 `tiny-yolo.weights` and the `tiny-yolo.cfg` for mobile devices

```
wget https://pjreddie.com/media/files/tiny-yolo.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo.cfg
```

### Convert a yolo model `.cfg + .weights` to a protobuf file `.pb` 
- It's useful, e.g. you can convert a yolo model and ues it as the input model for [TF Detect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java)
- The generated `.pb` are saved in `built_graph`

#### For YOLO V2
`python3 flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb --verbalise` 

#### For tiny YOLO V2
`python3 flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --savepb --verbalise` 

### Troubleshooting 

- The weight files from [Darknet](https://pjreddie.com/darknet/yolo/) may change their format and that may cause the tool to fail. Try to change the `self.WEIGHTS_HEADER_OFFSET` in the class `weights_walker` in `loader.py` to the discrepancy of the expected file size and the system reported file size.



### Reference
 - Prof. Andrew Ng's Convolutional Neural Networks [course](https://www.coursera.org/learn/convolutional-neural-networks). If you don't want to go through the `NOT-SO-EASY-TO-READ` papers of YOLO algorithm: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf)
 - [Darkflow repository](https://github.com/thtrieu) 
