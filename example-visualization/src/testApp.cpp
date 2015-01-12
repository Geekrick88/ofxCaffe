/*
 
 ofxCaffe - testApp.cpp
 
 The MIT License (MIT)
 
 Copyright (c) 2015 Parag K. Mital, http://pkmital.com
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 
 
 */

#include "testApp.h"
#include <memory>

//--------------------------------------------------------------
void testApp::setup(){
    
    width = 1280; height = 720;

    ofSetWindowShape(width, height);
    camera.initGrabber(width, height);
    color_img.allocate(width, height);
    
    current_model = 0;
    
    caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
    caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
    
    b_0 = b_1 = true;
    b_2 = b_3 = b_4 = false;
    
    b_mutex = false;
    
    layer_num = 0;
}

//--------------------------------------------------------------
void testApp::update(){
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    
    camera.update();
    color_img.setFromPixels(camera.getPixels());
    cv::Mat img(color_img.getCvImage()), img2;
    img.copyTo(img2);
    if(!b_mutex)
    {
        b_mutex = true;
        caffe->forward(img2);
        b_mutex = false;
    }
}

//--------------------------------------------------------------
void testApp::draw(){
    ofBackground(255);
    ofSetColor(255);
    if (b_0)
        color_img.draw(0, 0, width, height);
    
    if(!b_mutex)
    {
        b_mutex = true;
        if (caffe->getModelType() == ofxCaffe::OFXCAFFE_MODEL_BVLC_CAFFENET_34x17 ||
            caffe->getModelType() == ofxCaffe::OFXCAFFE_MODEL_BVLC_CAFFENET_8x8)
        {
            ofEnableAlphaBlending();
            ofSetColor(255, 255, 255, 200);
            caffe->drawDetectionGrid(width, height);
            ofDisableAlphaBlending();
        }
    
        ofDrawBitmapStringHighlight("model (-/+):  " + caffe->getModelTypeNames()[current_model], 20, 30);
        
        ofDisableAlphaBlending();
        if (b_1)
            caffe->drawLabelAt(20, 50);
        if (b_2)
            caffe->drawLayerXParams(0, 80, width, 32, layer_num);
        if (b_3)
            caffe->drawLayerXOutput(0, 80, width, 32, layer_num);
        if (b_4)
            caffe->drawProbabilities(0, 500, width, 200);
        b_mutex = false;
    }
    else
    {
        string s("Initializing new model...");
        ofDrawBitmapStringHighlight(s, width / 2.0 - s.length() / 2.0 * 8.0, height / 2.0);
    }

}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    if (key == '0')
        b_0 = !b_0;
    else if (key == '1')
        b_1 = !b_1;
    else if (key == '2')
        b_2 = !b_2;
    else if (key == '3')
        b_3 = !b_3;
    else if (key == '4')
        b_4 = !b_4;
    else if (key == '[')
    {
        layer_num = std::max<int>(0, layer_num - 1);
        cout << "layer_num '[' or ']': " << layer_num << endl;
    }
    else if (key == ']')
    {
        while (b_mutex) {} b_mutex = true;
        layer_num = std::min<int>(caffe->getTotalNumBlobs(), layer_num + 1);
        b_mutex = false;
        cout << "layer_num '[' or ']': " << layer_num << endl;
    }
    else if (key == '-' || key == '_')
    {
        current_model = (current_model == 0) ? (ofxCaffe::getTotalModelNums() - 1)  : (current_model - 1);
        while (b_mutex) {} b_mutex = true;
        caffe.reset();
        caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
        caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
        b_mutex = false;
    }
    else if (key == '+' || key == '=')
    {
        current_model = (current_model + 1) % ofxCaffe::getTotalModelNums();
        while (b_mutex) {} b_mutex = true;
        caffe.reset();
        caffe = std::shared_ptr<ofxCaffe>(new ofxCaffe());
        caffe->initModel(ofxCaffe::getModelTypes()[current_model]);
        b_mutex = false;
    }
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}
