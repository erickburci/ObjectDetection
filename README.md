![image](https://github.com/erickburci/ObjectDetection/assets/159087967/6a4c25a2-b24e-4924-8a56-ae33cbda46cb)

# Object Detection

In this project, I developed an object detector based on gradient features and sliding window classification. 

see ***ObjectDetection.ipynb*** for code

# Part 1 - Image Gradients

Write a function that takes a grayscale image as input and returns two arrays the same size as the image, the first of which contains the magnitude of the image gradient at each pixel and the second containing the orientation.

The function filters the image with simple x- and y-derivative filters that can compute the orientation and magnitude of the gradient vector at each pixel. 

![image](https://github.com/erickburci/ObjectDetection/assets/159087967/055eb29d-5c9b-4045-9d01-e85698d64b0d)
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/5cc5016b-ccc0-4565-8d92-dffa874c0e5b)

# Part 2 - Histograms of Gradient Orientations

The function below computes gradient orientation histograms over each 8x8 block of pixels in an image. The function bins the orientation into 9 equal sized bins between -pi/2 and pi/2. The input of the function will be an image of size HxW. The output is a three-dimensional array ***ohist*** whose size is (H/8)x(W/8)x9 where ***ohist[i,j,k]*** contains the count of how many edges of orientation k fell in block (i,j). If the input image dimensions are not a multiple of 8, I will use the function ***np.pad*** with the ***mode=edge*** option to pad the width and height up to the nearest integer multiple of 8.

To determine if a pixel is an edge, we need to choose some threshold. I suggest using a threshold that is 10% of the maximum gradient magnitude in the image. Since each 8x8 block will contain a different number of edges, we should normalize the resulting histogram for each block to sum to 1 (i.e., ***np.sum(ohist,axis=2)*** should be 1 at every  location).

Its best if the function loops over the orientation bins. For each orientation bin we'll need to identify those pixels in the image whose gradient magnitude is above the threshold and whose orientation falls in the given bin. We can do this easily in numpy using logical operations in order to generate an array the same size as the image that contains Trues at the locations of every edge pixel that falls in the given orientation bin and is above threshold. To collect up pixels in each 8x8 spatial block we can use the function ***ski.util.view_as_windows(...,(8,8),step=8)*** and ***np.count_nonzeros*** to count the number of edges in each block.

![image](https://github.com/erickburci/ObjectDetection/assets/159087967/13712101-3694-4680-94c6-e2c54b6188cd)
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/cbc2984c-240e-4514-b09e-57b92fdbea5d)

# Part 3 - Detection

Here I write a function that takes a template and an image and returns the top detections found in the image.

The function first computes the histogram-of-gradient-orientation feature map for the image, then correlates the template with the feature map. Since the feature map and template are both three dimensional, we will want to filter each orientation separately and then sum up the results to get the final response. If the image of size HxW then this final response map will be of size (H/8)x(W/8).

When constructing the list of top detections, the code implements non-maxima suppression so that it doesn't return overlapping detections. We can do this by sorting the responses in descending order of their score. Every time we add a detection to the list to return, we check to make sure that the location of this detection is not too close to any of the detections already in the output list. We can estimate the overlap by computing the distance between a pair of detections and checking that the distance is greater than say 70% of the width of the template.

![image](https://github.com/erickburci/ObjectDetection/assets/159087967/2fa1c463-f53c-4148-9828-7e7ca24820d4)

# Part 4 - Learning Templates

The final step is to implement a function to learn a template from positive and negative examples. The code takes a collection of cropped positive and negative examples of the object we are interested in detecting, extract the features for each, and generate a template by taking the average positive template minus the average negative template.

# Part 5 - Experiments

## Experiment 1: Face detection
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/3eee5982-dd2f-4657-a4d0-2f1a3f46737a)
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/cae3e37f-bd80-44b6-bc6f-34bf7d989e80)

## Experiment 2: Bottle detection
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/c6a19e3c-9506-46e4-a4f4-4ff00161e395)
![image](https://github.com/erickburci/ObjectDetection/assets/159087967/4a17c077-4f96-4564-bfbc-6bdd46aeda26)

## Discussion 
The detector worked well for the faces; it was able to match all 4 faces in the test image. I believe this is because faces are just detailed enough so that the histogram of gradient orientation is able to effectively differentiate between objects that are faces and objects that are not. Also, the test image background did not have a lot of noise (the bodies were the only distractors in the entire image). However, the detector did not work so well with the bottles. I believe this is because bottles have a very simple shape that can be easily replicated by many other objects that are not bottles. For example, the background of the bottle test image consisted of blurry circles which have curvature similar to that of the bottle neck taper. The detector may be improved if the mygradent function implemented a more complex method for constructing the x- and y-derivative filters. We were supposed to implement the simple method where we find the slope between two neighboring pixels, however, I think if we implemented the more sophisticated method where we first apply a gaussian filter (or use the function skimage.feature.canny) then perhaps the detector would have worked better.
