# ofxCaffe
## Interface for Caffe: Convolutional Architectures for Fast Feature Embedding from BVLC.  

![img0](https://github.com/pkmital/ofxCaffe/raw/master/img-0.png)
![img1](https://github.com/pkmital/ofxCaffe/raw/master/img-1.png)
![img2](https://github.com/pkmital/ofxCaffe/raw/master/img-2.png)
![img3](https://github.com/pkmital/ofxCaffe/raw/master/img-3.png)

Current Models
=============

* VGG ILSVRC 2014 (16 Layers): 1000 Object Categories
* VGG ILSVRC 2014 (19 Layers): 1000 Object Categories
* BVLC GoogLeNet: 1000 Object Categories
* Region-CNN ILSVRC 2013: 200 Object Categories (Region proposals not yet implemented)
* BVLC Reference CaffeNet: 1000 Object Categories
* BVLC Reference CaffeNet (Fully Convolutional) 8×8: 1000 Object Categories
* BVLC Reference CaffeNet (Fully Convolutional) 34×17: 1000 Object Categories
* MIT Places-CNN Hybrid (Places + ImageNet): 971 Object Categories + 200 Scene Categories = 1171 Categories

Instructions
============

(Warning: these probably won't work and will require edits/your help)

*  Install [Caffe](http://caffe.berkeleyvision.org/) and all dependencies (-lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lcblas)
*  Install [openFrameworks 64-bit](http://openframeworks.cc/download/) for osx I used this 1 year old branch which is about 1000 commits behind the master... I think the oF community is getting ready to release a stable 64-bit osx version??? [Nick Hardeman's OSX 64-bit openFrameworks branch](https://github.com/NickHardeman/openframeworks_osx_64)
*  clone this repo into of_directory/addons/ofxCaffe
*  clone [ofxOpenCv2461]() into of_directory/addons/ofxOpenCv2461 (could possibly be replaced by other opencv libraries, likely not the one that ships with openframeworks though; I've included OSX compiled OpenCV 2461 libraries in the addon/libs/opencv folder)
*  clone pkmMatrix into of_directory/../pkm/pkmMatrix (This is OSX only due to its depency on Accelerate.framework; provides vectorized operations; can be replaced with OpenCV; please submit pull request and I wiil merge...)
*  clone pkmHeatmap into of_directory/../pkm/pkmHeatmap (Converts grayscale images to RGB JET colormap using GPU)
*  Go to the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and download all necessary .caffemodel files into the bin/data directory (could make a script for this...; I know this last step is ridiculous.  You have to go into the code and see what the names are and I probably changed the names too... well some of them anyways.  I will make a script if there is enough interest...)

Example Project: Visualization
==============================

* '1': Toggle predicted label output 
* '2': Toggle layer parameters
* '3': Toggle layer outputs
* '4': Toggle probabilities graph
* '[' / ']': Change the current layer visualized
* '-' / '+': Change the current model
* '0': Toggle webcamera image


Troubleshooting
===============

* First make sure you can run Caffe and all tests (make runall)
* Check the Project.xcconfig defines and make sure they match up with where things should be (library files/source code)

To Do
======

* Properly crop images and mirror them to produce batch images
* R-CNN region proposals
* Possibly other models can support region proposals and still be fast
* Add Network in Network
* Add Flickr style Fine Tuning
* Alternatives to visualizing layers?  Deconvnets?  Projections?