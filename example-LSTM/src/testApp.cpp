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
    
    fbo.allocate(width, height);
    
    caffe = std::shared_ptr<ofxCaffeLSTM>(new ofxCaffeLSTM());
    caffe->initModel(ofxCaffeLSTM::getModelTypes()[ofxCaffeLSTM::OFXCAFFE_LSTM_MODEL_DEEP_LONG]);
    caffe->setSequenceLength(50);
    
    current_mode = TRAINING_MODE;
    
    b_mutex = false;
    
    class_label = -1.0;

}

//--------------------------------------------------------------
void testApp::update(){
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
    

}

//--------------------------------------------------------------
ofColor getColorForLabel(float label)
{
    label = ofMap(ofClamp(label, -1.0, 1.0), -1.0, 1.0, 0.0, 1.0);
    return ofColor(190  * label, 180 - label * 180, 190 * label);
}

//--------------------------------------------------------------
void testApp::draw(){
    ofBackground(0);
    ofEnableAntiAliasing();
    ofEnableAlphaBlending();
    ofEnableSmoothing();
    

    
    if(current_mode == TESTING_MODE)
    {
        ofSetColor(255);
        fbo.draw(0, 0, width, height);
        
        while(b_mutex) { }
        b_mutex = true;
        ofSetColor(140, 140, 180);
        ofSetLineWidth(5.0f);
        for (int idx_i = 1; idx_i < testing_labels.rows; idx_i++) {
        ofSetColor(getColorForLabel(class_label));
            ofDrawLine((0.5 + testing_labels.row(idx_i-1)[0]) * ofGetWidth(),
                       (0.5 + testing_labels.row(idx_i-1)[1]) * ofGetHeight(),
                       (0.5 + testing_labels.row(idx_i)[0]) * ofGetWidth(),
                       (0.5 + testing_labels.row(idx_i)[1]) * ofGetHeight());
        }
        b_mutex = false;
        
    }
    else
    {
        fbo.begin();
        
        ofBackground(0);
        
        // draw training data
        for (int example_i = 0; example_i < training_data.size(); example_i++) {
            
            ofSetLineWidth(1.0f);
            ofSetColor(getColorForLabel(training_data[example_i][0]));
            for (int idx_i = 1; idx_i < training_labels[example_i].rows; idx_i++) {
                ofDrawLine((0.5 + training_labels[example_i].row(idx_i-1)[0]) * ofGetWidth(),
                           (0.5 + training_labels[example_i].row(idx_i-1)[1]) * ofGetHeight(),
                           (0.5 + training_labels[example_i].row(idx_i)[0]) * ofGetWidth(),
                           (0.5 + training_labels[example_i].row(idx_i)[1]) * ofGetHeight());
                //            ofDrawBitmapString(ofToString((int)(training_labels[example_i].row(idx_i)[0] / training_labels[example_i][training_labels[example_i].rows-1] * 100)),
                //                               (0.5 + training_data[example_i].row(idx_i)[0]) * ofGetWidth(),
                //                               (0.5 + training_data[example_i].row(idx_i)[1]) * ofGetHeight());
            }
            ofSetColor(255);
            ofDrawBitmapString(ofToString(example_i+1),
                               (0.5 + training_labels[example_i].row(0)[0]) * ofGetWidth(),
                               (0.5 + training_labels[example_i].row(0)[1]) * ofGetHeight());
        }
        
        fbo.end();
        
        fbo.draw(0, 0, width, height);
    }
    
    ofDisableAntiAliasing();
    ofDisableAlphaBlending();
    ofDisableSmoothing();
    
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    cout << (char)key << endl;
    if(key == ' ')
    {
        if(current_mode == TESTING_MODE)
        {
            current_mode = TRAINING_MODE;
        }
        else
        {
            caffe->setBeginTraining();
            caffe->setTrainingData(training_data, training_labels);
            for(int i = 0; i < 1000; i++)
                caffe->doTrainingIteration();
            caffe->setBeginTesting();
            
            training_data.resize(0);
            training_labels.resize(0);
            
            current_mode = TESTING_MODE;
        }
    }
    else if(key == 'C')
    {
        caffe->setBeginTraining();
        for(int i = 0; i < 1000; i++)
            caffe->doTrainingIteration();
    }
    else if(key == 'T')
    {
        current_mode = current_mode == TESTING_MODE ? TRAINING_MODE : TESTING_MODE;
            
    }
    else if(key == 'D')
    {
        if(current_mode == TRAINING_MODE)
        {
            if(training_data.size())
            {
                training_data.erase(training_data.end());
                training_labels.erase(training_labels.end());
            }
        }
    }
    else if(key == '1')
    {
        class_label = -1.0;
        cout << "class-label: " << class_label << endl;
    }
    else if(key == '2')
    {
        class_label = 0.0;
        cout << "class-label: " << class_label << endl;
    }
    else if(key == '3')
    {
        class_label = 1.0;
        cout << "class-label: " << class_label << endl;
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
    if(current_mode == TRAINING_MODE)
    {
        float ex[2] = { x / (float)ofGetWidth() - 0.5, y / (float)ofGetHeight() - 0.5 };
        pkm::Mat m_ex(1, 2, ex);
        float l[1] = { class_label };
        pkm::Mat m_l(1, 1, l);
        training_example.addExample(m_l, m_ex);
    }
    else
    {
        while(b_mutex) { }
        b_mutex = true;
        
        float l[1] = { class_label };
        pkm::Mat input(1, 1, l), output;
        
        caffe->forward(input, output);
        testing_labels.push_back(output);
        b_mutex = false;
    }
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    if(current_mode == TRAINING_MODE)
    {
        training_example.reset();
        float ex[2] = { x / (float)ofGetWidth() - 0.5, y / (float)ofGetHeight() - 0.5 };
        pkm::Mat m_ex(1, 2, ex);
        float l[1] = { class_label };
        pkm::Mat m_l(1, 1, l);
        training_example.addExample(m_l, m_ex);
    }
    else
    {
        caffe->setBeginTesting();

        while(b_mutex) { }
        b_mutex = true;
        
        testing_data = pkm::Mat();
        testing_labels = pkm::Mat();
        
        float l[1] = { class_label };
        pkm::Mat input(1, 1, l), output;
        
        caffe->forward(input, output, true);
        testing_labels.push_back(output);
        b_mutex = false;
    }
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
    if(current_mode == TRAINING_MODE)
    {
        float ex[2] = { x / (float)ofGetWidth() - 0.5, y / (float)ofGetHeight() - 0.5 };
        pkm::Mat m_ex(1, 2, ex);
        float l[1] = { class_label };
        pkm::Mat m_l(1, 1, l);
        training_example.addExample(m_l, m_ex);
        training_data.push_back(training_example.getInterpolatedData(caffe->getSequenceLength()));
        training_labels.push_back(training_example.getInterpolatedLabels(caffe->getSequenceLength()));
    }
    else
    {
        while(b_mutex) { }
        b_mutex = true;
        
        float l[1] = { class_label };
        pkm::Mat input(1, 1, l), output;
        
        caffe->forward(input, output);
        testing_labels.push_back(output);
        b_mutex = false;
    }
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
