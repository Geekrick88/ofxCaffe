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

//--------------------------------------------------------------
#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCaffe.h"
#include "pkmMatrix.h"

//--------------------------------------------------------------
class trainingExample
{
public:
    //--------------------------------------------------------------
    trainingExample()
    {
        
    }
    
    //--------------------------------------------------------------
    void addExample(pkm::Mat ex, pkm::Mat l)
    {
        example.push_back(ex);
        labels.push_back(l);
    }
    
    //--------------------------------------------------------------
    void reset()
    {
        example = pkm::Mat();
        labels = pkm::Mat();
    }
    
    //--------------------------------------------------------------
    pkm::Mat & getData()
    {
        return example;
    }
    
    //--------------------------------------------------------------
    pkm::Mat & getLabels()
    {
        return labels;
    }
    
    //--------------------------------------------------------------
    pkm::Mat getInterpolatedData(size_t sequence_length)
    {
        pkm::Mat m_int;
        example.interpolate(sequence_length, example.cols, m_int);
        cout << "original :" << endl;
        example.print();
        cout << "interpolated: " << endl;
        m_int.print();
        return m_int;
    }
    
    //--------------------------------------------------------------
    pkm::Mat getInterpolatedLabels(size_t sequence_length)
    {
        pkm::Mat m_int;
        labels.interpolate(sequence_length, labels.cols, m_int);
        cout << "original :" << endl;
        labels.print();
        cout << "interpolated: " << endl;
        m_int.print();
        return m_int;
    }
    
private:
    //--------------------------------------------------------------
    pkm::Mat example; // N observations x D dimensions
    pkm::Mat labels;  // 1 x N observations
    
};

//--------------------------------------------------------------
class testApp : public ofBaseApp{

public:
    //--------------------------------------------------------------
    enum MODE {TESTING_MODE = 0, TRAINING_MODE = 1};
    
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
    // ptr to caffe obj
    std::shared_ptr<ofxCaffeLSTM> caffe;
    
    //--------------------------------------------------------------
    // window size
    int width, height;
    
    //--------------------------------------------------------------
    // storage for training data
    trainingExample training_example;
    vector<pkm::Mat> training_labels;
    vector<pkm::Mat> training_data;
    
    //--------------------------------------------------------------
    // storage for testing data
    pkm::Mat testing_data;
    pkm::Mat testing_labels;
    
    //--------------------------------------------------------------
    // training/testing
    MODE current_mode;
    
    ofFbo fbo;
    
    float class_label;
    
    bool b_mutex;
};
