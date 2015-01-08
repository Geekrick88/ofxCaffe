/*
 
 ofxCaffe.h
 
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

#include <cuda_runtime.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include "/usr/local/include/opencv2/core/core.hpp"
#include "/usr/local/include/opencv2/highgui/highgui.hpp"

#include <boost/algorithm/string.hpp>

#include "ofxCvMain.h"

#include "pkmMatrix.h"
#include "pkmHeatmap.h"

#include "ofMain.h"

using namespace caffe;
using namespace std;


class ofxCaffe {
    
public:
    typedef enum {
        OFXCAFFE_MODEL_VGG_16,
        OFXCAFFE_MODEL_VGG_19,
        OFXCAFFE_MODEL_HYBRID,
        OFXCAFFE_MODEL_BVLC_CAFFENET_34x17,
        OFXCAFFE_MODEL_BVLC_CAFFENET_8x8,
        OFXCAFFE_MODEL_BVLC_CAFFENET,
        OFXCAFFE_MODEL_BVLC_GOOGLENET,
        OFXCAFFE_MODEL_RCNN_ILSVRC2013,
        OFXCAFFE_MODEL_NOT_ALLOCATED
    } OFXCAFFE_MODEL_TYPE;
    
    ofxCaffe()
    :
    bAllocated(false)
    {   
        // Set GPU
        Caffe::set_mode(Caffe::GPU);
        int device_id = 0;
        Caffe::SetDevice(device_id);
        LOG(INFO) << "Using GPU";
        
        // Set to TEST Phase
        Caffe::set_phase(Caffe::TEST);
        
        model = OFXCAFFE_MODEL_NOT_ALLOCATED;
    }
    
    ~ofxCaffe()
    {
        if(bAllocated)
            delete net;
        
        for(int i = 0; i < layer1_imgs.size(); i++)
            delete layer1_imgs[i];
    }
    
    static vector<ofxCaffe::OFXCAFFE_MODEL_TYPE> getModelTypes()
    {
        vector<ofxCaffe::OFXCAFFE_MODEL_TYPE> d;
        d.push_back(OFXCAFFE_MODEL_VGG_16);
        d.push_back(OFXCAFFE_MODEL_VGG_19);
        d.push_back(OFXCAFFE_MODEL_HYBRID);
        d.push_back(OFXCAFFE_MODEL_BVLC_CAFFENET_34x17);
        d.push_back(OFXCAFFE_MODEL_BVLC_CAFFENET_8x8);
        d.push_back(OFXCAFFE_MODEL_BVLC_CAFFENET);
        d.push_back(OFXCAFFE_MODEL_BVLC_GOOGLENET);
        d.push_back(OFXCAFFE_MODEL_RCNN_ILSVRC2013);
        return d;
    }
    
    static vector<string> getModelTypeNames()
    {
        vector<string> d;
        d.push_back("VGG ILSVRC 2014 (16 Layers): 1000 Object Categories");
        d.push_back("VGG ILSVRC 2014 (19 Layers): 1000 Object Categories");
        d.push_back("MIT Places-CNN Hybrid (Places + ImageNet): 971 Object Categories + 200 Scene Categories = 1171 Categories");
        d.push_back("BVLC Reference CaffeNet (Fully Convolutional) 34x17: 1000 Object Categories");
        d.push_back("BVLC Reference CaffeNet (Fully Convolutional) 8x8: 1000 Object Categories");
        d.push_back("BVLC Reference CaffeNet: 1000 Object Categories");
        d.push_back("BVLC GoogLeNet: 1000 Object Categories");
        d.push_back("Region-CNN ILSVRC 2013: 200 Object Categories");
        return d;
    }
    
    static int getTotalModelNums()
    {
        return getModelTypes().size();
    }
    
    void initModel(OFXCAFFE_MODEL_TYPE model)
    {
        this->model = model;
        
        // Load pre-trained net (binary proto) and associated labels
        if (model == OFXCAFFE_MODEL_VGG_16) {
            net = new Net<float>(ofToDataPath("vgg-16.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("VGG_ILSVRC_16_layers.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_VGG_19) {
            net = new Net<float>(ofToDataPath("vgg-19.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("VGG_ILSVRC_19_layers.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_HYBRID) {
            net = new Net<float>(ofToDataPath("hybridCNN_deploy.prototxt"));
            net->CopyTrainedLayersFrom(ofToDataPath("hybridCNN_iter_700000.caffemodel"));
            loadHybridLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8) {
            net = new Net<float>(ofToDataPath("8x8-alexnet.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("bvlc_caffenet_full_conv.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17) {
            net = new Net<float>(ofToDataPath("34x17-alexnet.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("bvlc_caffenet_full_conv.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET) {
            net = new Net<float>(ofToDataPath("bvlc_reference_caffenet.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("bvlc_reference_caffenet.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_GOOGLENET) {
            net = new Net<float>(ofToDataPath("bvlc_googlenet.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("bvlc_googlenet.caffemodel"));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_RCNN_ILSVRC2013) {
            net = new Net<float>(ofToDataPath("bvlc_reference_rcnn_ilsvrc13.txt"));
            net->CopyTrainedLayersFrom(ofToDataPath("bvlc_reference_rcnn_ilsvrc13.caffemodel"));
            loadILSVRC2013();
        }
        else {
            cerr << "UNSUPPORTED MODEL!" << endl;
            OF_EXIT_APP(0);
        }
        
        
        width = net->input_blobs()[0]->width();
        height = net->input_blobs()[0]->height();
        
        // fcn:    B 104.00698793 G 116.66876762 R 122.67891434
        // vgg-16 : [103.939, 116.779, 123.68]
        // vgg-19 : [103.939, 116.779, 123.68]
        
        if (model == OFXCAFFE_MODEL_VGG_16 ||
            model == OFXCAFFE_MODEL_VGG_19)
        {
            mean_img = cv::Mat(cv::Size(width, height), CV_8UC3);
            cv::Vec3b mean_val;
            mean_val[0] = 103.939;
            mean_val[1] = 116.779;
            mean_val[2] = 123.68;
            for(int x = 0; x < width; x++)
                for(int y = 0; y < height; y++)
                    mean_img.at<cv::Vec3b>(x,y) = mean_val;
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8 ||
                 model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17 ||
                 model == OFXCAFFE_MODEL_BVLC_CAFFENET ||
                 model == OFXCAFFE_MODEL_HYBRID ||
                 model == OFXCAFFE_MODEL_BVLC_GOOGLENET ||
                 model == OFXCAFFE_MODEL_RCNN_ILSVRC2013)
        {
            if(model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8)
                dense_grid.allocate(8, 8);
            else if(model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17)
                dense_grid.allocate(34, 17);
        
            BlobProto blob_proto;
            if (model == OFXCAFFE_MODEL_HYBRID)
                ReadProtoFromBinaryFileOrDie(ofToDataPath("hybridCNN_mean.binaryproto").c_str(), &blob_proto);
            else
                ReadProtoFromBinaryFileOrDie(ofToDataPath("imagenet_mean.binaryproto").c_str(), &blob_proto);
            cout << "channels: " << blob_proto.channels() << endl;
            
            mean_img = cv::Mat(cv::Size(blob_proto.width(), blob_proto.height()), CV_8UC3);
            const unsigned int data_size = blob_proto.channels() * blob_proto.height() * blob_proto.width();
            
            for (int h = 0; h < blob_proto.height(); ++h) {
                uchar* ptr = mean_img.ptr<uchar>(h);
                int img_index = 0;
                for (int w = 0; w < blob_proto.width(); ++w) {
                    for (int c = 0; c < blob_proto.channels(); ++c) {
                        int datum_index = (c * blob_proto.height() + h) * blob_proto.width() + w;
                        ptr[img_index++] = blob_proto.data(datum_index);
                    }
                }
            }
            
            cv::resize(mean_img, mean_img, cv::Size(width, height));
        }
        
        bAllocated = true;
    }
    
    // Setting the image automatically calls forward prop and finds the best label
    void forward(cv::Mat& img)
    {
        // some of these networks are actually with 10 num, or 256 for hybrid, of different crops which I should figure out... then the original image is a bit larger.  for now i've simplified them to be just one figure... not sure how that effects performance...
        
        if(img.rows != height || img.cols != height)
            cv::resize(img, img, cv::Size(width, height));
        
        cv::cvtColor(img, img, CV_RGB2BGR);
        img = img - mean_img;
        
        //get datum
        Datum datum;
        CVMatToDatum(img, &datum);
        
//        vector<Datum> datum_vector;
//        datum_vector.push_back(datum);
//        
//        // Set vector for label
//        vector<int> labelVector;
//        labelVector.push_back(0);//push_back 0 for initialize purpose
//        
//        // Net initialization
//        float loss = 0.0;
//        boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
//        memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name("data"));
//        
//        memory_data_layer->AddDatumVector(datum_vector);
//        
//        // Run ForwardPrefilled
//        const vector<Blob<float>*>& result = net->ForwardPrefilled(&loss);
        
        //get the blob
        Blob<float> blob(1, datum.channels(), datum.height(), datum.width());
        
        //get the blobproto
        BlobProto blob_proto;
        blob_proto.set_num(1);
        blob_proto.set_channels(datum.channels());
        blob_proto.set_height(datum.height());
        blob_proto.set_width(datum.width());
        const int data_size = datum.channels() * datum.height() * datum.width();
        int size_in_datum = std::max<int>(datum.data().size(),
                                          datum.float_data_size());
        for (int i = 0; i < size_in_datum; ++i) {
            blob_proto.add_data(0.);
        }
        const string& data = datum.data();
        if (data.size() != 0) {
            for (int i = 0; i < size_in_datum; ++i) {
                blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
            }
        }
        
        //set data into blob
        blob.FromProto(blob_proto);
        
        //fill the vector
        vector<Blob<float>*> bottom;
        bottom.push_back(&blob);
        float type = 0.0;
        
        // Forward Prop
        const vector<Blob<float>*>& result = net->Forward(bottom, &type);
        
        // Get max label depending on architecture, just a single label
        if (model == OFXCAFFE_MODEL_VGG_16 ||
            model == OFXCAFFE_MODEL_VGG_19 ||
            model == OFXCAFFE_MODEL_BVLC_CAFFENET ||
            model == OFXCAFFE_MODEL_HYBRID ||
            model == OFXCAFFE_MODEL_BVLC_GOOGLENET ||
            model == OFXCAFFE_MODEL_RCNN_ILSVRC2013) {
            result_mat = pkm::Mat(result[0]->channels(), result[0]->height()*result[0]->width(), result[0]->cpu_data());
            result_mat.max(max, max_i);
            
            LOG(ERROR) << result[0]->num() << "x" <<  result[0]->channels() << "x" << result[0]->width() << "x" <<  result[0]->height() << " - max: " << max << " i " << max_i << " label: " << labels[max_i];
        }
        // or for dense architecture, many labels that can be summed/averaged/etc...
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8 ||
                 model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17)
        {
            pkm::Mat result_mat_grid(result[0]->channels(), result[0]->height()*result[0]->width(), result[0]->cpu_data());
            result_mat = result_mat_grid.sum(false);
            result_mat.max(max, max_i);
            pkm::Mat result_sliced;
            result_sliced = result_mat_grid.rowRange(max_i, max_i+1, false);
            unsigned long max_ixy;
            float max_sliced;
            result_sliced.max(max_sliced, max_ixy);
            max_x = max_ixy % result[0]->height();
            max_y = max_ixy / result[0]->width();
            result_sliced.divide(max_sliced);
            result_sliced.reshape(result[0]->height(), result[0]->width());
            
            cv::Mat d = result_sliced.cvMat();
            cv::Mat d2 = dense_grid.getCvImage();
            d.copyTo(d2);
            dense_grid.flagImageChanged();
            
            LOG(ERROR) << result[0]->width() << "x" <<  result[0]->height() << " - max: " << max << " i " << max_i << " label: " << labels[max_i];
        }
        else
        {
            cerr << "How did you get here?" << endl;
        }
    }
    
    void backward()
    {
        net->Backward();
    }
    
    OFXCAFFE_MODEL_TYPE getModelType()
    {
        return model;
    }
    
    // For dense models, draw an overlay representing the probabilities
    void drawDetectionGrid(int w, int h)
    {
        if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8 ||
            model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17)
            dense_grid.draw(0, 0, w, h);
    }
    
    // Draw the detected label
    void drawLabelAt(int x, int y)
    {
        ofDrawBitmapStringHighlight("label: " + labels[max_i], x, y);
        ofDrawBitmapStringHighlight("prob: " + ofToString(max), x, y+20);
    }
    
    int getTotalNumParamLayers()
    {
        return net->params().size();
    }
    
    // Draws the first network layer's parameters
    void drawLayerXParams(int px = 0,
                          int py = 60,
                          int width = 1280,
                          int images_per_row = 32,
                          int layer_num = 0)
    {
        boost::shared_ptr<Blob<float> > layer1 = net->params()[layer_num];
        string layer_name = net->layer_names()[layer_num];
        
        ostringstream oss;
        oss << layer1->num() << "x" << layer1->channels() << "x" << layer1->height() << "x" << layer1->width();
        cout << oss.str() << endl;
        
        while(layer1_imgs.size() < layer1->num())
        {
            ofxCvColorImage *img = new ofxCvColorImage();
            img->allocate(layer1->width(), layer1->height());
            layer1_imgs.push_back(img);
        }
        
        // go from Caffe's layout to many images in opencv
        // number N x channel K x height H x width W. Blob memory is row-major in layout so the last / rightmost dimension changes fastest. For example, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
        const float *fp_from = layer1->cpu_data();
        const int max_channels = std::min<int>(layer1->channels(), 3);
        for(int n = 0; n < layer1->num(); n++)
        {
            ofxCvColorImage *img = layer1_imgs[n];
            if(img->getWidth() != layer1->width() ||
               img->getHeight() != layer1->height())
                img->allocate(layer1->width(), layer1->height());
            
            unsigned char *fp_to = (unsigned char *)img->getCvImage()->imageData;
            cv::Mat to(img->getCvImage());
            int widthStep = img->getCvImage()->widthStep;
            for(int c = 0; c < max_channels; c++)
            {
                for(int w = 0; w < layer1->width(); w++)
                {
                    for(int h = 0; h < layer1->height(); h++)
                    {
                        fp_to[h * widthStep + 3 * w + c] = fp_from[ ((n * layer1->channels() + c) * layer1->height() + h) * layer1->width() + w ] * 255.0 + + 128.0;
                    }
                }
            }
            img->flagImageChanged();
            int nx = n % images_per_row;
            int ny = n / images_per_row;
            int padding = 1;
            int drawwidth = (width - padding * images_per_row) / (float)images_per_row;
            int drawheight = drawwidth;
            
            img->draw(px + nx * drawwidth + nx * padding + padding,
                      py + ny * drawheight + ny * padding + padding,
                      drawwidth, drawheight);
        }
        
        ofDrawBitmapStringHighlight("layer: " + layer_name + " " + oss.str() + " (only the first 3 channels are visualized as RGB)", px + 20, py + 20);
    }
    
    int getTotalNumBlobs()
    {
        return net->blobs().size();
    }
    
    // Draws the first network layer's output of the convolution.
    void drawLayerXOutput(int px = 0,
                          int py = 200,
                          int width = 1280,
                          int images_per_row = 32,
                          int layer_num = 1)
    {
        boost::shared_ptr<Blob<float> > layer1 = net->blobs()[layer_num];
        string layer_name = net->blob_names()[layer_num];
        ostringstream oss;
        oss << layer1->num() << "x" << layer1->channels() << "x" << layer1->height() << "x" << layer1->width();
        cout << oss.str() << endl;
        
        while(layer1_output_imgs.size() < layer1->channels())
        {
            ofxCvGrayscaleImage *img = new ofxCvGrayscaleImage();
            img->allocate(layer1->width(), layer1->height());
            layer1_output_imgs.push_back(img);
        }
        
        // number N x channel K x height H x width W. Blob memory is row-major in layout so the last / rightmost dimension changes fastest. For example, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
        const float *fp_from = layer1->cpu_data();
        for(int n = 0; n < layer1->num(); n++)
        {
            for(int c = 0; c < layer1->channels(); c++)
            {
                ofxCvGrayscaleImage *img = layer1_output_imgs[n];
                if(img->getWidth() != layer1->width() ||
                   img->getHeight() != layer1->height())
                    img->allocate(layer1->width(), layer1->height());
                unsigned char *fp_to = (unsigned char *)img->getCvImage()->imageData;
                cv::Mat to(img->getCvImage());
                int widthStep = img->getCvImage()->widthStep;
                
                for(int w = 0; w < layer1->width(); w++)
                {
                    for(int h = 0; h < layer1->height(); h++)
                    {
                        fp_to[h * widthStep + w] = fp_from[ ((n * layer1->channels() + c) * layer1->height() + h) * layer1->width() + w ] * 2.0;
                    }
                }
                img->flagImageChanged();
                int nx = (n*layer1->channels() + c) % images_per_row;
                int ny = (n*layer1->channels() + c) / images_per_row;
                int padding = 1;
                int drawwidth = (width - padding * images_per_row) / (float)images_per_row;
                int drawheight = drawwidth;

                ofPushMatrix();
                ofTranslate(px + nx * drawwidth + nx * padding,
                            py + ny * drawheight + ny * padding);
                
                cmap.begin(img->getTextureReference());
                img->draw(0, 0, drawwidth, drawheight);
                cmap.end();
                ofPopMatrix();
            }
        }
        
        ofDrawBitmapStringHighlight("layer: " + layer_name + " " + oss.str(), px + 20, py + 20);
    }
    
    void drawProbabilities(int px, int py, int w, int h)
    {
        ofPushMatrix();
        
        float padding = 10;
        ofSetColor(0, 0, 0, 180);
        
        ofRect(px, py, w, h);
        
        ofTranslate(px + padding, py + h - padding);
        
        ofSetColor(255);
        
        ofDrawBitmapStringHighlight("probabilities", 10, 10);
        
        float step = (w - 2.0 * padding) / (float)result_mat.size();
        float height_scale = (h - 2.0 * padding) / 1.0; //  / max
        for(int i = 1; i < result_mat.size(); i++)
        {
            ofLine((i - 1) * step, -result_mat[i-1] * height_scale,
                   i * step, -result_mat[i] * height_scale);
        }
        
        ofPopMatrix();
    }
    
    

    
private:
    
    OFXCAFFE_MODEL_TYPE model;
    
    // Load net
    Net<float> *net;
    
    // dense detection (8x8)
    ofxCvFloatImage dense_grid;
    
    // for drawing layers
    vector<ofxCvColorImage *> layer1_imgs;
    vector<ofxCvGrayscaleImage *> layer1_output_imgs;
    
    // for converting grayscale to rgb jet/cmaps...
    pkmColormap cmap;
    
    // Keep the mean for each forward pass
    cv::Mat mean_img;
    
    // probabilities of all classes
    pkm::Mat result_mat;
    
    // Value and Index of max
    float max = 0;
    unsigned long max_i = 0;
    unsigned short max_x, max_y;
    
    // necessary for input layer
    unsigned short width, height;
    
    // imagenet or hybrid labels
    vector<string> labels;
    
    // simple flag for when the model has been allocated
    bool bAllocated;
    
private:
    
    void loadImageNetLabels()
    {
        ofFile fp;
        fp.open(ofToDataPath("synset_words.txt"));
        ofBuffer buf;
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            string label;
            for(int i = 1; i < strs.size(); i++)
            {
                label += strs[i];
                label += " ";
            }
            labels.push_back(label);
        }
        
        fp.close();
    }
    
    void loadILSVRC2012()
    {
        ofFile fp;
        fp.open(ofToDataPath("ILSVRC2012.txt"));
        ofBuffer buf;
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            labels.push_back(line);
        }
        
        fp.close();
    }
    
    void loadILSVRC2013()
    {
        ofFile fp;
        fp.open(ofToDataPath("ILSVRC2013.txt"));
        ofBuffer buf;
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            labels.push_back(line);
        }
        
        fp.close();
    }
    
    void loadILSVRC2014()
    {
        ofFile fp;
        fp.open(ofToDataPath("ILSVRC2014.txt"));
        ofBuffer buf;
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            labels.push_back(line);
        }
        
        fp.close();
    }
    
    void loadHybridLabels()
    {
        map<string, string> imagenet_labels;
        ofFile fp;
        ofBuffer buf;
        
        fp.open(ofToDataPath("synset_words.txt"));
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            string label;
            for(int i = 1; i < strs.size(); i++)
            {
                label += strs[i];
                label += " ";
            }
            imagenet_labels[strs[0]] = label;
        }

        fp.close();
        
        fp.open(ofToDataPath("categoryIndex_hybridCNN.csv"));
        buf = fp.readToBuffer();
        while(!buf.isLastLine())
        {
            string line(buf.getNextLine());
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            cout << strs[0] << endl;
            if ( imagenet_labels.find(strs[0]) == imagenet_labels.end() ) {
                labels.push_back(strs[0]);
                cout << strs[0] << " - " << strs[1] << endl;
            } else {
                labels.push_back(imagenet_labels[strs[0]]);
                cout << imagenet_labels[strs[0]] << endl;
            }
        }
        
        fp.close();
    }
    
    
    void CVMatToDatum(cv::Mat& cv_img, Datum* datum) {
        CHECK(cv_img.depth() == CV_8U) <<
        "Image data type must be unsigned byte";
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->clear_data();
        datum->clear_float_data();
        int datum_channels = datum->channels();
        int datum_height = datum->height();
        int datum_width = datum->width();
        int datum_size = datum_channels * datum_height * datum_width;
        std::string buffer(datum_size, ' ');
        for (int h = 0; h < datum_height; ++h) {
            const uchar* ptr = cv_img.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < datum_width; ++w) {
                for (int c = 0; c < datum_channels; ++c) {
                    int datum_index = (c * datum_height + h) * datum_width + w;
                    buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            }
        }
        datum->set_data(buffer);
        return;
    }
};