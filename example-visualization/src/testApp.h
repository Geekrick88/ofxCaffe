/*
 
 ofxCaffe - testApp.h
 
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

#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCaffe.h"


class testApp : public ofBaseApp{

public:
    //--------------------------------------------------------------
    void setup();
    void update();
    void draw();
    
    //--------------------------------------------------------------
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
	
    //--------------------------------------------------------------
    // camera and opencv image objects
    ofQTKitGrabber camera;
    ofxCvColorImage color_img;
    
    //--------------------------------------------------------------
    // ptr to caffe obj
    std::shared_ptr<ofxCaffe> caffe;
    
    //--------------------------------------------------------------
    // which model have we loaded
    int current_model;
    
    //--------------------------------------------------------------
    // image and window dimensions
    int width, height;
    
    //--------------------------------------------------------------
    // simple flags for switching on drawing options of camera image/layers/parameters/probabilities
    bool b_0, b_1, b_2, b_3, b_4;
    
    //--------------------------------------------------------------
    // hacky mutex for when changing caffe model
    bool b_mutex;
    
    //--------------------------------------------------------------
    // which layer are we visualizing
    int layer_num;
};
