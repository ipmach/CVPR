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
