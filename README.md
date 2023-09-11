# pickleball_roomba
This is my ongoing effort to build a "roomba" to collect balls off a court. It's an undirected hobby project so I'll document the different tangents I go on below.

# Tangent (A): Vanishing point detection and line clustering
This is a classical computer vision approach to:
1. Detect two vanishing points in an image
2. Assign lines in the image to one of these two vanishing points

This could be a first step to automatically detect keypoints of a pickleball or tennis court in an image from an uncalibrated camera. I hope to use such a detector to label images scraped from the web, which I can use to train a CNN-based detector which will have much faster inference time.

See [readme](https://github.com/ekchapman/pickleball_roomba/tree/main/vanishing_points) for more details.

![02](https://github.com/ekchapman/pickleball_roomba/assets/43839555/19b343ce-0172-45f8-9326-8e8164eb71ea)

# Tangent (B): Pickleball vs Tennis scene classifier
A CNN-based classifier to distinguish between pickleball and tennis scenes. This one's just for fun; it's not really useful for a pickleball "roomba".

I built a RGB image classifier with pytorch, using a linear classifier atop a resnet50 tail as a feature extractor. I [scraped](https://github.com/YoongiKim/AutoCrawler) a few thousand images from Google Images and labeled them according to the search phrase: "[sport]", "professional [sport]", "people playing [sport]". The classifier is 81% accurate, struggling especially with heavily edited photos, logos, unfavorable cropping, mislabels, and other quirks of my primitive data collection process.

See [notebook](https://github.com/ekchapman/pickleball_roomba/blob/main/pb_vs_tennis_classifier/main.ipynb) for more details.

Below are 15 test images that I chose specifically to highlight the strengths and weaknesses of the model and dataset. Note some of the test images are mislabeled while some the model truly misclassifies. The labels are indicated by "p" or "t", and a classification that disagrees with the label is also marked with a "!".

![pb_vs_tennis](https://github.com/ekchapman/pickleball_roomba/assets/43839555/92e3140b-9234-4876-83ce-85f1fa23a12c)
