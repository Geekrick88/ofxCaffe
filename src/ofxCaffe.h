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

#include "ofxOpenCv.h"

#include "pkmMatrix.h"
#include "pkmHeatmap.h"

#include "ofMain.h"

using namespace caffe;
using namespace std;

#define WITH_OFXCAFFE_LSTM

#ifdef WITH_OFXCAFFE_LSTM

class ofxCaffeLSTM
{
public:
    typedef enum {
        OFXCAFFE_LSTM_MODEL_DEEP_LONG,
        OFXCAFFE_LSTM_MODEL_DEEP_SHORT,
        OFXCAFFE_LSTM_MODEL_SHALLOW_LONG,
        OFXCAFFE_LSTM_MODEL_SHALLOW_SHORT,
        OFXCAFFE_LSTM_MODEL_NOT_ALLOCATED
    } OFXCAFFE_LSTM_MODEL_TYPE;
    
    ofxCaffeLSTM()
    :
    b_allocated(false), b_set_training_data(false)
    {
        // Set GPU
        Caffe::set_mode(Caffe::GPU);
        int device_id = 0;
        Caffe::SetDevice(device_id);
        LOG(INFO) << "Using GPU";
        
        // Set to TEST Phase
        Caffe::set_phase(Caffe::TEST);
        
        model = OFXCAFFE_LSTM_MODEL_NOT_ALLOCATED;
    }
    
    ~ofxCaffeLSTM()
    {
        if(b_allocated)
        {
            
        }
    }
    
    static vector<ofxCaffeLSTM::OFXCAFFE_LSTM_MODEL_TYPE> getModelTypes()
    {
        vector<ofxCaffeLSTM::OFXCAFFE_LSTM_MODEL_TYPE> d;
        d.push_back(OFXCAFFE_LSTM_MODEL_DEEP_LONG);
        d.push_back(OFXCAFFE_LSTM_MODEL_DEEP_SHORT);
        d.push_back(OFXCAFFE_LSTM_MODEL_SHALLOW_LONG);
        d.push_back(OFXCAFFE_LSTM_MODEL_SHALLOW_SHORT);
        return d;
    }
    
    static vector<string> getModelTypeNames()
    {
        vector<string> d;
        d.push_back("LSTM Deep Long");
        d.push_back("LSTM Deep Short");
        d.push_back("LSTM Shallow Long");
        d.push_back("LSTM Shallow Short");
        return d;
    }
    
    static int getTotalModelNums()
    {
        return getModelTypes().size();
    }
    
    
    void initModel(OFXCAFFE_LSTM_MODEL_TYPE model)
    {
        this->model = model;
        string net_solver;
        sequence_length = 10;
        if (model == OFXCAFFE_LSTM_MODEL_DEEP_LONG)
        {
            net_solver = string("../../../../../addons/ofxCaffe/models/deep_lstm_long_solver.prototxt");
        }
        else if (model == OFXCAFFE_LSTM_MODEL_DEEP_SHORT)
        {
            net_solver = string("../../../../../addons/ofxCaffe/models/deep_lstm_short_solver.prototxt");
        }
        else if (model == OFXCAFFE_LSTM_MODEL_SHALLOW_LONG)
        {
            net_solver = string("../../../../../addons/ofxCaffe/models/shallow_lstm_long_solver.prototxt");
        }
        else if (model == OFXCAFFE_LSTM_MODEL_SHALLOW_SHORT)
        {
            net_solver = string("../../../../../addons/ofxCaffe/models/shallow_lstm_short_solver.prototxt");
        }
        else
        {
            
        }
        
        caffe::SolverParameter solver_param;
        caffe::ReadProtoFromTextFileOrDie(ofToDataPath(net_solver, true).c_str(), &solver_param);
        
        solver.reset(GetSolver<float>(solver_param));
        net_train = solver->net();
        net_test = solver->test_nets()[0];
        
        solver->PreSolve();
        
    }
    
    // addTrainingExample...: each row as as sequence...
    // train()
    
    double f_x(double t) {
        return 0.5*sin(2*t) - 0.05*cos(17*t + 0.8)
        + 0.05*sin(25*t+10) - 0.02*cos(45*t + 0.3);
    }
    
    void setTrainingData(vector<pkm::Mat> training_data, vector<pkm::Mat> training_labels)
    {
        
        batch_size = training_data.size();
        sequence_length = training_data[0].rows;
        num_input_channels = training_data[0].cols;
        num_label_channels = training_labels[0].cols;
        
        for (int i = 0; i < training_data.size(); ++i) {
//            training_data[i].zNormalizeEachCol();
            cout << "training data " << i << ":" << endl;
            training_data[i].print();
            
            vector<Datum> d;
            vector<vector<float> > l;

            for (int j = 0; j < batch_size; ++j) {
                Datum datum;
                datum.set_channels(num_input_channels);
                datum.set_width(1);
                datum.set_height(1);
                for (int k = 0; k < num_input_channels; k++)
                    datum.add_float_data(training_data[j].row(i)[k]);
                d.push_back(datum);
                vector<float> tmp;
                for (int k = 0; k < num_label_channels; k++)
                    tmp.push_back(training_labels[j].row(i)[k]);
                l.push_back(tmp);
            }

            batch_data.push_back(d);
            batch_labels.push_back(l);
        }
        
        layers = net_train->layers();
        layer_param = layers[0]->layer_param();
        
        b_set_training_data = true;
    }
    
    void setNumChannels(size_t ch)
    {
        num_input_channels = ch;
    }
    
    void setSequenceLength(size_t sz)
    {
        sequence_length = sz;
    }
    
    size_t getSequenceLength()
    {
        return sequence_length;
    }
    
    void makeTrainingData()
    {
        cout << "training data" << endl;
        const unsigned char input = 0;
        datum.set_channels(num_input_channels);
        datum.set_width(1);
        datum.set_height(1);
        datum.set_data(&input, 1);
        
        
        // Get max to scale data to lie on [-1, 1]
        float mean = 0;
        float max_abs = 0;
        for (int i = 0; i < sequence_length; ++i) {
            float val = f_x(i * 0.01);
            max_abs = max(max_abs, abs(val));
        }
        
        // Get mean
        for (int i = 0; i < sequence_length; ++i) {
            mean += f_x(i * 0.01) / max_abs;
        }
        mean /= sequence_length;
        
        // Make t
        for (int i = 0; i < sequence_length; ++i) {
            vector<float> l;
            float y = f_x(i*0.01) / max_abs - mean;
            l.push_back(y);
            data.push_back(datum);
            labels.push_back(l);
        }
        
        layers = net_train->layers();
        layer_param = layers[0]->layer_param();
        batch_size = layer_param.memory_data_param().batch_size();
        CHECK_EQ(sequence_length % batch_size, 0) << "sequence length should be divided by batchsize";
        
        
        for (int i = 0; i < sequence_length / batch_size; ++i) {
            vector<Datum> d;
            vector<vector<float> > l;
            
            for (int j = 0; j < batch_size; ++j) {
                d.push_back(datum);
                int idx = i * batch_size + j;
                vector<float> tmp;
                tmp.push_back(labels[idx].at(0));
                l.push_back(tmp);
            }
            
            batch_data.push_back(d);
            batch_labels.push_back(l);
        }
        
        b_set_training_data = true;
    }
    
    void setBeginTraining()
    {
        smoothed_loss = 0;
        Caffe::set_phase(Caffe::TRAIN);
        iter = 0;
    }
    
    float doTrainingIteration()
    {
        if (b_set_training_data)
        {
            int batch_idx = iter % (sequence_length / batch_size);
            
            vector<Datum>& batch_d = batch_data[batch_idx];
            vector<vector< float> >& batch_l = batch_labels[batch_idx];
            
            ((SeqMemoryDataLayer<float>*)layers[0].get())->DataFetch(batch_d, batch_l, batch_idx == 0);
            solver->SolveIter(smoothed_loss, losses);
            
            iter++;
        }
        
        return smoothed_loss;
    }
    
    void doTraining()
    {
        setBeginTraining();
        
        while(!solver->IsFinished()) {
            doTrainingIteration();
        }
    }
    
    void setBeginTesting()
    {
        Caffe::set_phase(Caffe::TEST);
        net_test->ShareTrainedLayersWith(net_train.get());
    }
    
    void doTesting()
    {
        // Output Test
        
        const vector<boost::shared_ptr<Layer<float> > >& test_layers = net_test->layers();
        for (int i = 0; i < sequence_length; ++i) {
            ((SeqMemoryDataLayer<float>*)test_layers[0].get())->DataFetch(datum, i == 0);
            const vector<Blob<float>* >& result = net_test->Forward(bottom);
            CHECK_EQ(result.size(), 1);
            const float* output = result[0]->cpu_data();
            CHECK_EQ(result[0]->count(), 1);
//            vector<float>& l = labels[i];
//            cout << l[0] << " " << output[0] << endl;
            cout << output[0] << endl;
        }
    }
    
    // calls forward prop and finds the best label
    void forward(pkm::Mat& input, pkm::Mat &output, bool b_begining_of_sequence = false)
    {
        Datum datum;
        datum.set_channels(num_input_channels);
        datum.set_width(1);
        datum.set_height(1);
        for (int k = 0; k < num_input_channels; k++)
            datum.add_float_data(input[k]);
        
        const vector<boost::shared_ptr<Layer<float> > >& test_layers = net_test->layers();
        ((SeqMemoryDataLayer<float>*)test_layers[0].get())->DataFetch(datum, b_begining_of_sequence);

        const vector<Blob<float>* >& result = net_test->Forward(bottom);
        const float* output_ptr = result[0]->cpu_data();
        output = pkm::Mat(1, result[0]->count(), output_ptr);
//        cout << "num: " << result[0]->num() << " channels: " << result[0]->channels() << " width: " << result[0]->width() << " height: " << result[0]->height() << endl;

//        sequence.print();
        output.print();
    }
    
    
private:
    // Which model is loaded
    OFXCAFFE_LSTM_MODEL_TYPE model;
    
    // Load net
    boost::shared_ptr<Net<float> > net_train, net_test;
    
    // Training layers
    vector<boost::shared_ptr<Layer<float> > > layers;
    LayerParameter layer_param;
    
    // Training batch data
    vector<vector<Datum> > batch_data;
    vector<vector<vector<float> > > batch_labels;
    
    // Solver
    boost::shared_ptr<Solver<float> > solver;
    
    vector<Datum> data;
    vector<vector<float> > labels;
    vector<float> losses;
    
    vector<Blob<float>* > bottom;
    
    // current training iteration
    size_t iter;
    float smoothed_loss;
    
    Datum datum;
    
    // Input parameter's sequence length, batch size, and channels
    size_t sequence_length, batch_size, num_input_channels, num_label_channels;
    
    // simple flag for when the model has been allocated
    bool b_allocated;
    
    // is there data to train on
    bool b_set_training_data;
};

#endif

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
    b_allocated(false)
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
        if(b_allocated)
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
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/vgg-16.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/VGG_ILSVRC_16_layers.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_VGG_19) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/vgg-19.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/VGG_ILSVRC_19_layers.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_HYBRID) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/hybridCNN_deploy.prototxt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/hybridCNN_iter_700000.caffemodel", true));
            loadHybridLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_8x8) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/8x8-alexnet.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_caffenet_full_conv.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET_34x17) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/34x17-alexnet.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_caffenet_full_conv.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_CAFFENET) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_reference_caffenet.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_reference_caffenet.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_BVLC_GOOGLENET) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_googlenet.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_googlenet.caffemodel", true));
            loadImageNetLabels();
        }
        else if (model == OFXCAFFE_MODEL_RCNN_ILSVRC2013) {
            net = new Net<float>(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_reference_rcnn_ilsvrc13.txt", true));
            net->CopyTrainedLayersFrom(ofToDataPath("../../../../../addons/ofxCaffe/models/bvlc_reference_rcnn_ilsvrc13.caffemodel", true));
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
                ReadProtoFromBinaryFileOrDie(ofToDataPath("../../../../../addons/ofxCaffe/models/hybridCNN_mean.binaryproto", true).c_str(), &blob_proto);
            else
                ReadProtoFromBinaryFileOrDie(ofToDataPath("../../../../../addons/ofxCaffe/models/imagenet_mean.binaryproto", true).c_str(), &blob_proto);
            cout << "channels: " << blob_proto.channels() << endl;
            
            mean_img = cv::Mat(cv::Size(blob_proto.width(), blob_proto.height()), CV_8UC3);
            const size_t data_size = blob_proto.channels() * blob_proto.height() * blob_proto.width();
            
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
        
        b_allocated = true;
    }
    
    pkm::Mat getLayerByName(string name = "conv5", bool b_collapse_data = true)
    {
        const boost::shared_ptr<Blob<float> >& result = net->blob_by_name(name);
        pkm::Mat result_mat;
        const float *fp_from = result->cpu_data();
        
        for (size_t n = 0; n < result->num(); n++)
        {
            for (size_t c = 0; c < result->channels(); c++)
            {
                pkm::Mat fp_to(result->height(), result->width());
                size_t widthStep = result->width();
                
                for(size_t w = 0; w < result->width(); w++)
                {
                    for(size_t h = 0; h < result->height(); h++)
                    {
                        fp_to[h * widthStep + w] = fp_from[ ((n * result->channels() + c) * result->height() + h) * widthStep + w ] * 2.0;
                    }
                }
                
                if(b_collapse_data)
                {
                    float m = pkm::Mat::mean(fp_to.data, fp_to.size());
                    result_mat.push_back(m);
                }
                else
                    result_mat.push_back(fp_to);
            }
        }
        return result_mat;
        
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
        
        //get the blob
        Blob<float> blob(1, datum.channels(), datum.height(), datum.width());
        
        //get the blobproto
        BlobProto blob_proto;
        blob_proto.set_num(1);
        blob_proto.set_channels(datum.channels());
        blob_proto.set_height(datum.height());
        blob_proto.set_width(datum.width());
        const int data_size = datum.channels() * datum.height() * datum.width();
        size_t size_in_datum = std::max<int>(datum.data().size(),
                                          datum.float_data_size());
        for (size_t i = 0; i < size_in_datum; ++i) {
            blob_proto.add_data(0.);
        }
        const string& data = datum.data();
        if (data.size() != 0) {
            for (size_t i = 0; i < size_in_datum; ++i) {
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
            size_t max_ixy;
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
    void drawLayerXParams(size_t px = 0,
                          size_t py = 60,
                          size_t width = 1280,
                          size_t images_per_row = 32,
                          size_t layer_num = 0)
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
        const size_t max_channels = std::min<int>(layer1->channels(), 3);
        for(size_t n = 0; n < layer1->num(); n++)
        {
            ofxCvColorImage *img = layer1_imgs[n];
            if(img->getWidth() != layer1->width() ||
               img->getHeight() != layer1->height())
                img->allocate(layer1->width(), layer1->height());
            
            unsigned char *fp_to = (unsigned char *)img->getCvImage()->imageData;
            cv::Mat to(img->getCvImage());
            int widthStep = img->getCvImage()->widthStep;
            for(size_t c = 0; c < max_channels; c++)
            {
                for(size_t w = 0; w < layer1->width(); w++)
                {
                    for(size_t h = 0; h < layer1->height(); h++)
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
                      py + ny * drawheight + ny * padding + padding + 20,
                      drawwidth, drawheight);
        }
        
        ofDrawBitmapStringHighlight("layer: " + layer_name + " " + oss.str() + " (only the first 3 channels are visualized as RGB)", px + 20, py + 10);
    }
    
    int getTotalNumBlobs()
    {
        return net->blobs().size();
    }
    
    // Draws the first network layer's output of the convolution.
    void drawLayerXOutput(size_t px = 0,
                          size_t py = 200,
                          size_t width = 1280,
                          size_t images_per_row = 32,
                          size_t layer_num = 1)
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
        
        ofSetColor(255);
        
        // number N x channel K x height H x width W. Blob memory is row-major in layout so the last / rightmost dimension changes fastest. For example, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
        const float *fp_from = layer1->cpu_data();
        for(size_t n = 0; n < layer1->num(); n++)
        {
            for(size_t c = 0; c < layer1->channels(); c++)
            {
                ofxCvGrayscaleImage *img = layer1_output_imgs[n];
                if(img->getWidth() != layer1->width() ||
                   img->getHeight() != layer1->height())
                    img->allocate(layer1->width(), layer1->height());
                unsigned char *fp_to = (unsigned char *)img->getCvImage()->imageData;
                cv::Mat to(img->getCvImage());
                int widthStep = img->getCvImage()->widthStep;
                
                for(size_t w = 0; w < layer1->width(); w++)
                {
                    for(size_t h = 0; h < layer1->height(); h++)
                    {
                        fp_to[h * widthStep + w] = fp_from[ ((n * layer1->channels() + c) * layer1->height() + h) * layer1->width() + w ] * 2.0;
                    }
                }
                img->flagImageChanged();
                size_t nx = (n*layer1->channels() + c) % images_per_row;
                size_t ny = (n*layer1->channels() + c) / images_per_row;
                size_t padding = 1;
                size_t drawwidth = (width - padding * images_per_row) / (float)images_per_row;
                size_t drawheight = drawwidth;

                ofPushMatrix();
                ofTranslate(px + nx * drawwidth + nx * padding,
                            py + ny * drawheight + ny * padding + 20);
                
                cmap.begin(img->getTexture());
                img->draw(0, 0, drawwidth, drawheight);
                cmap.end();
                ofPopMatrix();
            }
        }
        
        ofDrawBitmapStringHighlight("layer: " + layer_name + " " + oss.str(), px + 20, py + 10);
    }
    
    void drawGraph(const pkm::Mat &mat, string title, size_t px, size_t py, size_t w, size_t h, float scale = 1.0f)
    {
        ofPushMatrix();
        
        float padding = 10;
        ofSetColor(0, 0, 0, 180);
        
        ofDrawRectangle(px, py, w, h);
        
        ofTranslate(px + padding, py + h - padding);
        
        ofSetColor(255);
        
        ofDrawBitmapStringHighlight(title, 10, 10);
        
        float step = (w - 2.0 * padding) / (float)mat.size();
        
        
        float height_scale = (h - 2.0 * padding) / scale;
        for(size_t i = 1; i < mat.size(); i++)
        {
            ofDrawLine((i - 1) * step, -mat[i-1] * height_scale,
                   i * step, -mat[i] * height_scale);
        }
        
        ofPopMatrix();
    }
    
    void drawProbabilities(size_t px, size_t py, size_t w, size_t h)
    {
        drawGraph(result_mat, "probabilities", px, py, w, h, 1.0f);
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
    size_t max_i = 0;
    size_t max_x, max_y;
    
    // necessary for input layer
    size_t width, height;
    
    // imagenet or hybrid labels
    vector<string> labels;
    
    // simple flag for when the model has been allocated
    bool b_allocated;
    
private:
    
    void loadImageNetLabels()
    {
        ofFile fp;
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/synset_words.txt", true));
        ofBuffer buffer(fp);
        for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
            string line = *it;
            cout << line << endl;
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            string label;
            for(size_t i = 1; i < strs.size(); i++)
            {
                label += strs[i];
                label += " ";
            }
            labels.push_back(label);
        }
        
        fp.close();
        
        cout << "[ofxCaffe]:: Read " << labels.size() << " labels." << endl;
    }
    
    void loadILSVRC2012()
    {
        ofFile fp;
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/ILSVRC2012.txt", true));
        ofBuffer buffer(fp);
        for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
            string line = *it;
            labels.push_back(line);
        }
        
        fp.close();
        
        cout << "[ofxCaffe]:: Read " << labels.size() << " labels." << endl;
    }
    
    void loadILSVRC2013()
    {
        ofFile fp;
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/ILSVRC2013.txt", true));
        ofBuffer buffer(fp);
        for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
            string line = *it;
            labels.push_back(line);
        }
        
        fp.close();
        
        cout << "[ofxCaffe]:: Read " << labels.size() << " labels." << endl;
    }
    
    void loadILSVRC2014()
    {
        ofFile fp;
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/ILSVRC2014.txt", true));
        ofBuffer buffer(fp);
        for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
            string line = *it;
            labels.push_back(line);
        }
        
        fp.close();
        
        cout << "[ofxCaffe]:: Read " << labels.size() << " labels." << endl;
    }
    
    void loadHybridLabels()
    {
        map<string, string> imagenet_labels;
        ofFile fp;
        ofBuffer buf;
        
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/synset_words.txt", true));
        ofBuffer buffer(fp);
        for (ofBuffer::Line it = buffer.getLines().begin(), end = buffer.getLines().end(); it != end; ++it) {
            string line = *it;
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("\t "));
            string label;
            for(size_t i = 1; i < strs.size(); i++)
            {
                label += strs[i];
                label += " ";
            }
            imagenet_labels[strs[0]] = label;
        }

        fp.close();
        
        fp.open(ofToDataPath("../../../../../addons/ofxCaffe/models/categoryIndex_hybridCNN.csv", true));
        ofBuffer buffer2(fp);
        for (ofBuffer::Line it = buffer2.getLines().begin(), end = buffer2.getLines().end(); it != end; ++it) {
            string line = *it;
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
        
        cout << "[ofxCaffe]:: Read " << labels.size() << " labels." << endl;
    }
    
    
    void CVMatToDatum(cv::Mat& cv_img, Datum* datum) {
        CHECK(cv_img.depth() == CV_8U) <<
        "Image data type must be unsigned byte";
        datum->set_channels(cv_img.channels());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->clear_data();
        datum->clear_float_data();
        size_t datum_channels = datum->channels();
        size_t datum_height = datum->height();
        size_t datum_width = datum->width();
        size_t datum_size = datum_channels * datum_height * datum_width;
        std::string buffer(datum_size, ' ');
        for (size_t h = 0; h < datum_height; ++h) {
            const uchar* ptr = cv_img.ptr<uchar>(h);
            size_t img_index = 0;
            for (size_t w = 0; w < datum_width; ++w) {
                for (size_t c = 0; c < datum_channels; ++c) {
                    size_t datum_index = (c * datum_height + h) * datum_width + w;
                    buffer[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            }
        }
        datum->set_data(buffer);
        return;
    }
};