# Tangent (A): Vanishing point detection and hirozontal / vertical line classification
See [`vanishing_points/main.py`](https://github.com/ekchapman/pickleball_roomba/blob/main/vanishing_points/main.py) for a script that detects vanishing points in an RGB image and simultaneously assigns detected lines to one or the other vanishing point.

## Motivation
I want a deep vision model to detect pickleball court keypoints (intersections of court lines) but alas I have no training data and don't want to label by hand. So (A) is part of an effort to automatically label images scraped from the web. Importantly, no calibration data is available for these images. (A) detects "horizontal" and "vertical" court lines, from which one could probably find keypoints by using RANSAC to map straight lines to known pickleball court lines. That's a work in progress, made more difficult by the lack of intrinsic calibration data, which makes some techniques like homography and undistortion less viable.

## Performance
(A) Takes 10-30 seconds to run and strikes a tough balance between sensitivity and outlier rejection. Note that it ignores vertical lines (marked in red). Here are a few examples from images off the web, with all tunable parameters set to default:

![02](https://github.com/ekchapman/pickleball_roomba/assets/43839555/19b343ce-0172-45f8-9326-8e8164eb71ea)

![01](https://github.com/ekchapman/pickleball_roomba/assets/43839555/24a97160-4637-407b-9e76-a17784000ab5)

![03](https://github.com/ekchapman/pickleball_roomba/assets/43839555/c6b66ce5-4c45-415e-ab3f-88323c3feddb)

## Approach
I take a multi-step approach that uses common CV and optimization techniques in sequence.
1. Gaussian blur
2. Canny edge detection
3. Hough line detection
4. Agglomerative clustering, then "fusing" clusters into one segment.
5. Simultaneous line classification and vanishing point regression.

First consider the original image...

![original](https://github.com/ekchapman/pickleball_roomba/assets/43839555/16cbbbd3-a89f-4035-967f-1aecba7c5f39)

### Gaussian blur
To remove noisy edge detections, especially from fences, nets, trees, and people. Unfortunately this can blur out a court line too. Perhaps I could look into [edge-presering smoothing](https://en.wikipedia.org/wiki/Edge-preserving_smoothing). 

![blurred](https://github.com/ekchapman/pickleball_roomba/assets/43839555/2a669390-9e7c-4e35-a213-33905cce25bf)

### Canny edge detection

![edges](https://github.com/ekchapman/pickleball_roomba/assets/43839555/1a40e19e-5e15-478e-a38a-ec1bbe156997)

### Hough line detection

![hough](https://github.com/ekchapman/pickleball_roomba/assets/43839555/dbb5efaa-9f2f-4410-bbaf-76fcfd9c2ac7)

### Agglomerative clustering, "fusing" clusters.

Agglomerative clustering with single linkage clusters line segments together based on similarity. If one segment is similar to another in the cluster, it gets added. In my case, I computed (dis)similarity by a weighted sum of their angular distance and minimum-possible linear distance.

For each cluster, a "fused" segment is generated. The angle is determined by averaging the angles of all segments, weighted by their length. The endpoints of the "fused" segment is determined by the length of the two most distant endpoints of the cluster when all endpoints are projected onto an axis at the same angle as the "fused" segment.

![grouped](https://github.com/ekchapman/pickleball_roomba/assets/43839555/87ee7f9d-ba21-45ea-951a-ad8a79670571)

### Simultaneous line classification and VP regression

I define a minimization problem with `scipy.optimize.minimize` where the cost function inputs the following:
1. Two vanishing points
2. Weights that assign each line (except vertical lines, which are omitted) to one of the two vanishing points.

Consider the following diagram where $p_k$ is a vanishing point $k$ and $c_i$ is the centerpoint of a segment $i$ with endpoints $a_i$ and $b_i$. We want choose vanishing point $p_k$ to minimize $\alpha_{ki} = r \theta$, or failing that, assign segment $i$ to the other vanishing point such that $\alpha_{ki}$ will not be counted towards the cost.

<img src="https://github.com/ekchapman/pickleball_roomba/assets/43839555/174d3918-5c43-4e7e-b569-42671d2b9aea" width="300" height="140">

More formally, we want to minimize:

$$\sum_{i=1}^N \alpha_{1i}^2 w_i^2 + \sum_{i=1}^N \alpha_{2i}^2 \left( 1 - w_i \right)^2 $$

Given two vanishing points $p_1$ and $p_2$ and $N$ non-vertical line segments.

Local minema can sometimes be a problem, but I had success by choosing sensible initial estimates. I initialized $p_1$ and $p_2$ to be on opposing sides of the image and halfway down along the y-axis. I initialized $w=\vec{0.5}$.

The weights converge at either $0$ or $1$. So line $i$ belongs to $p_1$ if $w_i > 0.5$ and belongs to $p_2$ if $w_i < 0.5$.

![02](https://github.com/ekchapman/pickleball_roomba/assets/43839555/4ca0849a-aaf3-4cc1-bbd0-dcedb646acc0)
