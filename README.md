# pickleball_roomba
This is my ongoing effort to build a "roomba" to collect balls off a court. It's an undirected hobby project so I'll document the different tangents I go on below.

# Tangent (A): Vanishing point detection and line clustering
This is a classical computer vision approach to:
1. Detect two vanishing points in an image
2. Assign lines in the image to one of these two vanishing points

This could be a first step to automatically detect keypoints of a pickleball or tennis court in an image from an uncalibrated camera. I hope to use such a detector to label images scraped from the web, which I can use to train a CNN-based detector which will have much faster inference time.

See [readme](https://github.com/ekchapman/pickleball_roomba/tree/main/vanishing_points) for more details.

![02](https://github.com/ekchapman/pickleball_roomba/assets/43839555/19b343ce-0172-45f8-9326-8e8164eb71ea)
