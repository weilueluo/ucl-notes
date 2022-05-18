# COMP0026 Image Processing

https://github.com/LouAgapito/ImPro26/

## Digital Image

> TODO: week1

## Segmentation

Finite set of non-overlapping regions that covers the image.

Using Gestalt Factors makes intuitive sense but difficult to analysis algorithmically.

### Thresholding

Binary classification using $I(x,y) > \text{threshold}$. Can be global or local adaptive. It is good when:

- low level of noise
- objects of same class have similar intensity
- homogeneous lighting

It does not consider the context / pixel connectivity.
<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814585.png" alt="image-20220517200944297" style="zoom:50%;" />

### Connected Components

```python
# grouping binary image's 1 components
# connected component is depth first search
def connected_components(b_im): 
    label_im = zeros_like(b_im)
    curr_label = 1
    for (x, y) in b_im:
        if label_im[x, y] == 0 \ # not yet labeled
                and b_im[x, y]: # and it is one of the components to group
            label(x, y, b_im, label_im, curr_label)  # then we label this region
            curr_label += 1  # increment, move on to next region
        
# recursively find neighbors and label them as the given current label
def label(x, y, b_im, label_im, curr_label):
    label_im[x, y] = curr_label
    for nx, ny in neighbour(x, y):
        if label_im[x, y] == 0 \ # not yet labeled
        		and b_im[x, y]: # and it is one of the components to group
            label(nx, ny, label_im, curr_label)
```

4-component neighbors may generate error, compared to 8-component neighbors.

This method does work work well for edges with disconnection.

Connected components is also part of the initial Vincent and Soille's watershed segmentation algorithm.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814631.png" alt="image-20220517201031651" style="zoom:50%;" />

### Region Growing

```python
# region grow is breath first search
def region_grow(im, visited, seed):
    # im: image
    # visited: number of times each pixels had been visited, visited(p) > 0 means it belongs to a region
    # seed: the initial value of this region
    queue = Queue([seed])
    while queue is not empty:
        pixel = queue.pop()
        if we should include this pixel in the current region:
            visited(pixel) += 1
            for x, y in neighbors(pixel):
                if visited((x, y)) == 0:
                    queue.add((x, y))
                
```

- Issues
  - Seed points need to be suitable.
  - Minimum area threshold need to be suitable.
  - Similarity threshold need to be suitable.
  - More image on the adjacent pixels is better.
- Advantages
  - Can separate regions according to properties we define (needs to be suitable).
  - Good result if the original image have clear edges.
  - We can determine the seed and it will find the whole region.
  - We can use multiple criteria for growing the region.
  - It is efficient as we can bound the number of pixels access theoretically.
- Disadvantages
  - Unless thresholded, a continuous path of points related to color may exist, which connects any two points in the image.
  - Practically random memory access slows down the algorithms, requires adaption.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814654.png" alt="image-20220517201054536" style="zoom:50%;" />

### Watershed

```python
def watershed(gray_im, seeds):
    # Meyer's flooding algorithm
    label = im.copy().fill(-1)
    p_queue = PriorityQueue()
    for seed in seeds:
        label[seed] = unique_label()
        for x, y in neighbors(seed):
            p_queue.add(x, y, priority=gray_im[x, y]) # use gray level as priority
    while p_queue is not empty:
        x, y = p_queue.pop_min()
        for nx, ny in neighbors(x, y): # assign xy's label to neighbor
            if label[nx, ny] == -1:  # not labeled yet
                label[nx, ny] = label[x, y]
            	p_queue.add((nx, ny), priority=gray_im[nx, ny])
```

It can produce nice result on smooth transitioning edges.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814693.png" alt="image-20220517201133138" style="zoom:50%;" />

But it can produce over-segmented results.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814722.png" alt="image-20220517201202644" style="zoom:50%;" />

By choosing a good set of markers, this can greatly enhance the result.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652814782.png" alt="image-20220517201301993" style="zoom:50%;" />

### Analysis

- Thresholding
  - Relies on a single global threshold
  - Produces separate regions (may prefer connected components)
- Region growing
  - Supervised, requires 1+ seed point
  - Relies on a single threshold
  - Produces one region
- Watershed
  - May be supervised or unsupervised
  - No need for threshold parameter
  - Partitions image into many regions
  - To get good results best to specify number and positions of seeds

### Clustering / Statistical Methods

#### K-means

> TODO: explain K means

- Advantages
  - Simple and fast
  - Converges to a local minimum of the distance function
- Disadvantages
  - Need to pick K
  - Sensitive to initialization
  - Sensitive to outliers
  - Only finds “spherical” clusters

#### Means-shift

```python
def mean_shift(im, curr_center, window):
    # choose a window and location as hyperparameters
    # compute mean of the data in window
    # center the search window to the new mean location
    # repeat until convergence
    # optional: merge window of similar center
    last_center = compute_mean(im, center, window)
    while last_center != curr_center:
        last_center = curr_center
        curr_center = compute_mean(im, center, window)

def compute_mean(im, center, window):
    sum_pos = [0, 0]
    total = 0
    for x, y in pixels(center, window):
        sum_pos += im[x, y]
        total += 1
    return sum_pos / total
```

- Advantages
  - Robust to outlier
  - Does not assume spherical clusters
  - Just a single parameters, windows size
  - Able to find variable number of modes
- Disadvantages
  - Output depends on window size
  - Computationally expensive
  - Does not scale well with high dimension

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652816074.png" alt="image-20220517203434070" style="zoom:50%;" />

#### Watershed VS K-Means VS Mean-shift

- Watershed
  - May be supervised or unsupervised 
  - No need for threshold parameter
  - Partitions image into many regions 
  - To get good results best to specify number and positions of seeds
- K-Means 
  - Unsupervised
  - Works in high-dimensions 
  - May produce disjoint segments
  - Must specify number of clusters but not position
- Mean-shift
  - Unsupervised
  - Works in high-dimensions (kind of)
  - May produce disjoint segments
  - “Only” specify windows size

### Graph-based Methods

#### Greedy Merging

1. Initialize each pixel as separate node
2. Define edge weight (e.g. pixel difference)
3. Connect node by edge weight in ascending order until some threshold, and merge as we go.
   - A better way to merge: two segments are merged if the link between them is smaller than any link within these two segments + $\tau(C)=k/|C|$, where $C$ is the component (segment), $k$ is the total number of nodes (pixels), and $|C|$ is number of nodes in the component.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652817355.png" alt="image-20220517205555335" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220517_1652817382.png" alt="image-20220517205622642" style="zoom:50%;" />

It is very fast, but sensitive to noise and may greedily chooses too large regions.

#### Normal Cuts

> TODO: ++ pg 97, week 2 image segmentation part 2

#### Splitting

> TODO: ++

### Evaluation

#### ROC

Used to measure performance of binary classifier

- sensitivity: true positive rate
- 1-specificity: false positive rate

ROC = a graph that is plotted with true positive rate against false positive rate as its threshold varies

Area under curve (AUC) closer to 1 = better

## Transformation

### Linear

For grayscale image, we can use linear transformation as contrast stretch

-  $f(x)=\alpha x+\beta$
- I think $\alpha$ is like contrast and $\beta$ is like brightness.

A function $L$ is linear operation if:
$$
L\left(\alpha I_{1}+\beta I_{2}\right)=\alpha L\left(I_{1}\right)+\beta L\left(I_{2}\right)
$$

- i.e. additive and homogeneous

| <img src="C:/Users/etsun/AppData/Roaming/Typora/typora-user-images/image-20220518105135848.png" alt="image-20220518105135848" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867518.png" alt="image-20220518105157980" style="zoom:50%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867536.png" alt="image-20220518105216819" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867553.png" alt="image-20220518105232989" style="zoom:50%;" /> |



### Non-linear

For grayscale image, this can be used for gamma correction (corrects luminance)

- $f(x)=A x^{\gamma}$.
  - where $A=255^{1-\gamma}$.
  - $\gamma < 1$ encoding gamma / gamma compression.
  - $\gamma > 1$ decoding gamma / gamma expansion.

| <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867610.png" alt="image-20220518105329958" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867623.png" alt="image-20220518105343615" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867662.png" alt="image-20220518105422103" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867691.png" alt="image-20220518105451863" style="zoom:50%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



### Histogram

We can apply histogram equalization by ensuring all values are used equally often:
$$
h(v)=\operatorname{round}\left(\frac{c d f(v)-c d f_{\min }}{(M \times N)-c d f_{\min }} \times(L-1)\right)
$$

- $M$ and $N$ are the image dimensions
- $cdf_{min}$ is the minimum non-zero value of the cumulative distribution function ($n$ pixels against each grayscale)
- $L$ is the number of gray level used

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652867456.png" alt="image-20220518105054521" style="zoom:50%;" />



### Affine

$$
x^{\prime}=a x+b y+t_{x}\\
y^{\prime}=c x+d y+t_{y}\\
\left(\begin{array}{l}x^{\prime} \\ y^{\prime}\end{array}\right)=\left(\begin{array}{ll}a & b \\ c & d\end{array}\right)\left(\begin{array}{l}x^{\prime} \\ y^{\prime}\end{array}\right)+\left(\begin{array}{l}t_{x} \\ t_{y}\end{array}\right)
$$

linear transformation: translate, scale, sheer, and rotate.

```python
def warp_image(im, transform):
    inv_trans = inverse(transform)  # always use inverse transformation
    out = zeros_like(im)
    for x, y in out.pixels:
        out[x, y] = im[inv_trans(x, y)]
    return out
```

Note the `inv_trans` may not result in integer coordinate, we need to do some intepolation

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652869027.png" alt="image-20220518111707882" style="zoom:50%;" />

### Polynomial Warp

We may also want to use quadratic transformation, which allows us to bend lines in the images.
$$
x^{\prime}=a_{0}+a_{1} x+a_{2} y+a_{3} x^{2}+a_{4} x y+a_{5} y^{2}\\
y^{\prime}=b_{0}+b_{1} x+b_{2} y+b_{3} x^{2}+b_{4} x y+b_{5} y^{2}
$$
<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220518_1652869184.png" alt="image-20220518111944712" style="zoom:50%;" />

### Controlled Point Warp

It is pretty hard to define a correct quadratic transformation, however, we can move some points to specific locations, interpolate the displacement at intermediate positions and generate a polynomial warp.
$$
\begin{gathered}
\left(\begin{array}{cc}
x_{1}^{\prime} & y_{1}^{\prime} \\
x_{2}^{\prime} & y_{2}^{\prime} \\
\vdots & \vdots \\
x_{m}^{\prime} & y_{m}^{\prime}
\end{array}\right)=\left(\begin{array}{cccccc}
1 & x_{1} & y_{1} & x_{1}^{2} & x_{1} y_{1} & y_{1}^{2} \\
1 & x_{2} & y_{2} & x_{2}^{2} & x_{2} y_{2} & y_{2}^{2} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
1 & x_{m} & y_{m} & x_{m}^{2} & x_{m} y_{m} & y_{1}^{2}
\end{array}\right)\left(\begin{array}{ccc}
a_{0} & b_{0} \\
a_{1} & b_{1} \\
a_{2} & b_{2} \\
a_{3} & b_{3} \\
a_{4} & b_{4} \\
a_{5} & b_{5}
\end{array}\right) \\
A=X P
\end{gathered}
$$
Least squares estimate of $P$ is $(m \geq 6)$:
$$
P=\left(X^{T} X\right)^{-1} X^{T} A
$$

#### Applications

- Special effects
  - Film industry
  - Computer games
- Image registration (transform different sets of data into one coordinate system, e.g. aligning image with translation and rotation)
  - Medical imaging
  - Security

### Morphing

For face morphing:

1. Find correspondences, i.e. corresponding face landmarks.
   - (Optional) Add additional correspondences at image corners / edges.
2. Triangulate.
3. Generate intermediate triangles at different time $t$.
4. For each pair of triangles for each intermediate frame
   1. Calculate the transformation (inverse)
   2. Apply the affine warp transformation to generate in-between images

## Filtering

- It is a technique for modifying or enhancing an image. 
- It is a low-level (pixel) processing operations
- Filter: let intensity through selectively

Uses:

- Noise reduction
- Smoothing
- Feature enhancement



### Spatial

#### Convolution

Apply a kernel $K$ to the image repeatedly
$$
\begin{aligned}
I^{\prime}(x, y) &=\sum_{i=-a}^{a} \sum_{j=-b}^{b} K(i, j) I(x-i, y-j) \\
&=\sum_{i=-a}^{a} \sum_{j=-b}^{b} K(-i,-j) I(x+i, y+j)
\end{aligned}
$$

$$
K(i, j)=K(-i,-j) \quad \Leftrightarrow \quad \text {convolution } \equiv \text { correlation}
$$

Correlation is convolution with kernel rotated by 180 degree. This makes no difference if the kernel is symmetric like Gaussian and Laplacian, but not kernel like derivative. Convolution is just multiplication in frequency domain, which is generally associative, which allows for pre-convolving and save computation. Correlation however, is multiple with complex conjugate in frequency domain, which is not associative.

#### Smoothing

> low-pass filter

like Gaussian kernel, Box kernel

Gaussian kernel is separable, which means we can e.g. separate a (3x3) Gaussian kernel into two (1x3 and 3x1) Gaussian kernel, performing two convolution to achieve the same effect as the original kernel. The benefit is that we can reduce computation

> TODO: add gaussian kernel formula

- Advantages
  - Rotationally symmetric
  - Has a single lobe/mode - Neighbour's influence decreases monotonically
  - Still one lobe in frequency domain - No corruption from high frequencies
  - Simple relationship to σ
  - Easy to implement efficiently

#### Derivative

#### Laplacian of Gaussian

### Frequency

#### Fourier Transform

#### Magnitude & Phase

#### Fourier Basis Functions

#### Image Spectra

#### Convolution Theorem

## Sampling / Image Pyramids / Blending

### Sampling & Aliasing

### Gaussian Pyramid

### Laplacian Pyramid

### Blending

## Edge Detection

### Gradient

### Edge Map

### Laplacian / Difference of Gaussian

### Laplacian Zero Crossing

### Hough Transform

## Corner Detection

### Auto Correlation

### Harris Corner Detection

### Harris Matrix Eigenvalues

### SIFT

### Susan / FAST

## Image Matching

### Template Matching

###  SSD / NCC

## Optical Flow

### Brightness Consistency

### Equation

### Aperture

### Lucas-Kanade Algorithm