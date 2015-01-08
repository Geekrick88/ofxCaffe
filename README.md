# ofxCaffe
## Interface for Caffe: Convolutional Architectures for Fast Feature Embedding from BVLC.  

!(https://github.com/pkmital/ofxCaffe/raw/master/img-0.png)
!(https://github.com/pkmital/ofxCaffe/raw/master/img-1.png)
!(https://github.com/pkmital/ofxCaffe/raw/master/img-2.png)
!(https://github.com/pkmital/ofxCaffe/raw/master/img-3.png)
!(https://github.com/pkmital/ofxCaffe/raw/master/img-4.png)
!(https://github.com/pkmital/ofxCaffe/raw/master/img-5.png)

### Instructions
================

(Warning: these probably won't work and will require edits/your help)

*  Install [Caffe](http://caffe.berkeleyvision.org/) and all dependencies (-lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lcblas)
*  Install [openFrameworks](http://openframeworks.cc/download/)
*  clone this repo into of_directory/addons/ofxCaffe
*  clone [ofxOpenCv2461]() into of_directory/addons/ofxOpenCv2461 (could possibly be replaced by other opencv libraries, likely not the one that ships with openframeworks though; I've included OSX compiled OpenCV 2461 libraries in the addon/libs/opencv folder)
*  clone pkmMatrix into of_directory/../pkm/pkmMatrix
*  clone pkmHeatmap into of_directory/../pkm/pkmHeatmap
*  Go to the Caffe Model Zoo and download all necessary .caffemodel files into the bin/data directory (could make a script for this...)


### Troubleshooting
===================

* First make sure you can run Caffe and all tests (make runall)
* Check the Project.xcconfig defines and make sure they match up with where things should be (library files/source code)

### To Do
=========

* Properly crop images and mirror them to produce batch images
* R-CNN region proposals
* Possibly other models can support region proposals and still be fast
* Add Network in Network
* Add Flickr style Fine Tuning