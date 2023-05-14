# Object-detection using YoLo
<img src="https://miro.medium.com/max/1400/1*bSLNlG7crv-p-m4LVYYk3Q.png" width="400px" height="150px">

# Mentors
* Jayavarapu Surya Abhinai
* Sushree Shivani Sethi
# Mentees
* Aakarsh Bansal
* Raajan
* Tejas S
* Shriya Sheri
# Aim 
The project aims to to build yolo architecture to detect objects 

# Libraries used
* Numpy
* Pandas
* Pytorch
* Matplotlib
* OpenCv

## 1. What is YOLO?<br>
Yolo is a state-of-the-art, object detection algorithm. It was developed by Joseph Redmon. The biggest advantage over other architectures is that speed without much compramising in accuracy. The Yolo model family models are really fast, much faster than R-CNN and others.In this article we will go through the third version of yolo yolov3 its architecture, how it works and its implementation in pytorch.

## 2. How Yolo Works? <br>
​
As yolo works with one look at an image,sliding window is not a good approach for it.Instead the entire image can be splitted in to grids of size SxS cells.Now each cell will be responsible for predicting different things.
​
Typically, as is the case for all object detectors the features learned by the convolutional layers are passed onto a classifier/regressor which makes the detection prediction.ie,coordinates of the bounding boxes, the class label.. etc but in YOLO, the prediction is done by using a convolutional layer which uses 1 x 1 convolutions.
​
The important thing to note is that the output is a feature map.Each cell of these feature map predicts different things.Let me explain it in detail.Each cell will predict 4 coordinates for bounding box,probability of bounding box and class probabilities for each of the class.Here probability of bounding box means probability that a box contains an object. In case that there is no object in some grid cell, it is important that confidence value is very low for that cell.For COCO dataset we have 80 classes.So in total each cell will predict 85 values.
​
In Yolo, anchor boxes are used to predict bounding boxes. The main idea of anchor boxes is to predefine two different shapes. They are called anchor boxes or anchor box shapes. In this way, we will be able to associate two predictions with the two anchor boxes. In general, we might use even more anchor boxes (five or even more). Anchors were calculated on the COCO dataset using k-means clustering. 
​
So, we have (B x (5 + C)) entries in the feature map. B represents the number of bounding boxes each cell can predict. Each of the bounding boxes have 5 + C attributes, which describe the center coordinates, the dimensions, the bounding box probability score and C class probability for each bounding box. YOLO v3 predicts 3 bounding boxes for every cell. ie, technically each cell have 3 anchors.therefore 3 bounding boxes.

<img src="https://blog.paperspace.com/content/images/2018/04/yolo-5.png" width="500px" height="500px">

For each bounding box, We have<br>
* 4 cordinates (tx,ty,tw,th)<br>
* Probability of bounding box. ie,probability that an object is present inside the bounding box<br>
* Class probabilities for each class (Here, 80 classes)<br>

If there is some offset from the top left corner by cx,cy, then the predictions correspond to:

* **bx = σ(tx) + cx**<br>

* **by = σ(ty) + cy**<br>

* **bw = pw x e^tw**<br>

* **bh = ph x e^th**<br>

Here bx, by, bw, bh are the x,y center co-ordinates, width and height of our prediction. tx, ty, tw, th are the network outputs. cx and cy are the top-left co-ordinates of the grid. pw and ph are anchors dimensions for the box.Before going further we will explain about the centre cordinates.


### Centre coordinates and boundingbox dimensions
Actually we are running centre coordinate prediction through a sigmoid function. so the value of the output to be between 0 and 1. Why should this be the case? Did you ever thought about that.

Normally yolo doesn't directly predict the bounding boxes.It predicts the offsets relative to the top left corner of the grid cell which is predicting the object. Also it is normalised by the dimensions of the cell from the feature map, which is, 1.

For example, consider the case of our dog image. If the prediction for center is (0.5, 0.3).Then the  actual centre cordinate lies at (6.5,6.3) in 13x13 feature map.similarly what if the prediction for center is (1.4,0.9) then the actual centre cordinate lies at (7.4,6.9).This is opposite to concept of YOLO. That is why the output is passed through a sigmoid function so it becomes in range 0 to 1.

The dimensions of the bounding box are predicted by applying a log space transform to output and then multiplying with an anchor.The resultant predictions, bw and bh, are normalised by the height and width of the anchors.So, if the predictions bx and by for the box containing the dog are (0.7, 0.8), then the actual width and height on if we have anchor box of height 10 and width 8 is is (10 x 0.7, 8 x 0.8).

## Predictions

In one pass we can go from an input image to the output tensor which corresponds to the detections for the image. Also, it is worth mentioning that YOLOv3 predicts boxes at 3 different scales.In the above figure it is represented by y1,y2 and y3.

Using the COCO dataset, YOLOv3 predicts 80 different classes. YOLO outputs bounding boxes and class prediction as well. If we split an image into a 13 x 13 grid of cells and use 3 anchors box, the total output prediction is 13 x 13 x 3 or 169 x 3. However, YOLOv3 uses 3 different prediction scales which splits an image into (13 x 13), (26 x 26) and (52 x 52) grid of cells and with 3 anchors for each scale. So, the total output prediction will be((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10647 bounding boxes.

<img src="https://s3-eu-west-1.amazonaws.com/ppreviews-plos-725668748/15596051/preview.jpg" width="1000px" height="500px">

## Non-Max suppression

Suppose N is the threshold and we have two lists A and B.Then for each class detected in image, 

* Select the proposal with highest confidence score, remove it from A and add it to the final proposal list B. (Initially B is empty).
* Now compare this proposal with all the proposals by calculating the IOU (Intersection over Union) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from A.
* Again take the proposal with the highest confidence from the remaining proposals in A and remove it from A and add it to B.
* Once again calculate the IOU of this proposal with all the proposals in A and eliminate the boxes which have high IOU than threshold.
* This process is repeated until there are no more proposals left in A.


<img src="https://miro.medium.com/max/1400/1*6d_D0ySg-kOvfrzIRwHIiA.png" width="800px" height="300px">

### References:

https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/util.py <br>
https://www.ijeat.org/wp-content/uploads/papers/v8i3S/C11240283S19.pdf
http://dspace.srmist.edu.in/jspui/bitstream/123456789/34874/1/P9689.pdf
