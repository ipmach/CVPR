# CVPR

I create this repository to study Computer Vision from my master's. The examples implemented are from the book "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods. The code has different sections:

- Geometric Transformations
- Value Transformations
- Canny Edge Detection
- Harris Stephens Corner Detector
- Hough Transform

All the examples are in the different notebooks. Feel free to use this repository to understand better Computer Vision's basic functions and for educational purposes. The code is not optimized for real applications.

## Code

The main code can be found in the ComputerVision folder.

### Geometric Transformations

The geometric transformations change the shape and position of the figure of the image.

<img src="https://github.com/ipmach/CVPR/blob/main/img/watches.PNG" alt="drawing" width="600"/>

#### Geometric matrices

``` python
class GeometricTransformations:
   def identity(): pass
   def scaling_k(cx, cy): pass
   def translation_k(tx, ty): pass
   def rotation_k(angle): pass
   def vertical_shear_k(lambda_): pass
   def horizontal_shear_k(lambda_): pass
````
 
 #### Interpolation 

``` python
class Interpolations:
  def no_inter(img, M, N, x, y): pass 
  def bilinear(img, M, N, x, y): pass
  def nearest_neighbour(img, M, N, x, y): pass
````

### Mapping

``` python
def inverse_mapping(img, A, default_value=255,
                    interpolation=Interpolations.bilinear, scale=(1, 1)): pass
def forward_mapping(img, A, default_value=255, scale=(1, 1)): pass
```

### Value Transformations 

The value transformations change the intensity of the pixels.

#### Value Transformations using filters 

Applying kernel filters.


<img src="https://github.com/ipmach/CVPR/blob/main/img/pipes.PNG" alt="drawing" width="600"/>

##### Filter

``` python
class Filter:
   def add_bounds(img, kernel_size): pass
   def remove_bounds(img, kernel_size): pass
   def correlation(img, g): pass
   def convolution(img, g): pass
   def apply_filter(img, g, method='convolution'): pass 
   def apply_operators(img, gx, gy, method='convolution'): pass
   def correlation_l(img, r, c): pass
   def convolution_l(img, r, c): pass
   def apply_filter_l(img, r, c, method='convolution'): pass
   def apply_operators_l(img, gx_r, gx_c, gy_r,
                       gy_c, method='convolution'): pass
````

##### Filter Kernel

``` python
def box_kernel(m): pass
def gaussian_kernel(m, gamma): pass 
def laplacian_kernel(alternative=False): pass
def prewitt_operators(): pass
def sobel_operators(): pass 
def box_kernel_l(m): pass
def gaussian_kernel_l(m, gamma): pass
def prewitt_operators_l(): pass
def sobel_operators_l(): pass
````


#### Value Transformations using histogram 

Applying histogram transformations.


<img src="https://github.com/ipmach/CVPR/blob/main/img/histogram.PNG" alt="drawing" width="600"/>

##### Histogram

``` python
class Histogram:
   def calculate_cdf(count): pass
   def plot_histogram(hist, cdf=True, figsize=(15, 5)): pass
   def histogram(img, visualize=True, normalize=False,
                  full=False, L=255, figsize=(15, 5)): pass
   def equalization_book(img, L=255, round_=True): pass
   def equalization_wiki(img, L=255): pass 
   def equalization_prof(img, L=255): pass
   def equalization(img, L=255, method='prof'): pass
   def matching(img1, img2): pass
````

