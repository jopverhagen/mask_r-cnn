<h1>Object detection using a region based Convolutional Neural Network  (R-CNN)</h1>

[![youtube rcnn test](youtube.png)](https://www.youtube.com/watch?v=OVc69ptx_N0)

<img src="output.png">

Turns out, it’s not that hard doing object detection with opencv and numpy only.
Even on a CPU it does a reasonable job for single images (like the example code). For live streaming (as seen in the youtube vid) GPU processing is required.

This model is trained on the MSCOCO dataset (backbone architecture InceptionV2). As you can see, it does a good job detecting me as a person(97%) while wearing a helmet. Although it’s not so sure if i’m on a bicycle (55%) or motorcycle(61%)

paper: https://arxiv.org/pdf/1703.06870.pdf

<h2>Usage</h2>

<ul>
  <li>git clone https://github.com/jopverhagen/mask_r-cnn.git</li>
  <li>Download pretrained weights from: http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz. (place frozen_inference_graph.pb in main folder)</li>

  <li>input.jpg: Your input image</li>
  <li> run: python run_rcnn.py</li>
  
 </ul>
