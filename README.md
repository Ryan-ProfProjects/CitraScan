# CitraScan-Pipeline
A hardware-agnostic computer vision pipeline designed to democratize early-warning detection of Huanglongbing (HLB) for farmers across varying field conditions.

Imagine your entire livelihood is on the line because a deadly disease is destroying your orange grove. You pull out your smartphone to use a disease-detection app, but the AI fails because the image is blurry, the lighting is poor, or the leaf isn’t perfectly framed. Today’s agricultural AI is often trained on clean ideal images that don’t reflect the messy conditions of real farms. At CitraScan, we are providing open-source data so that AI models can learn to perform reliably in the field. Our mission is to ensure cutting-edge diagnostics belong to the farmers who need it most, regardless of the smartphone in their pocket.

The CitraScan pipeline evaluates incoming data with an unsupervised algorithmic data reliability governor. Multiple metrics are leveraged to account for blurriness, resolution, lighting conditions, and symptom positioning in a structured Reliability Vector. 

### Blurriness

Blurriness is measured through Laplacian variance that detects how much the max/min(s) (optimum(s)) of gradients vary in the image. Thinking of the image as a gradient space, a local pixel patch of high laplacian value is around a certian optimum. In a sharp image, the transitions from lighter to darker intensity happen over a very small spatial scale, meaning the optimums are smaller. If an image has more of these optimums, its pixel gradients change most rapidly (high second gradient/derivative) and has more sharpness by carrying more detail as a result. Blurrier or less sharp images has less of these optimums because local pixel patches stay more uniform and require larger patches to observe a similiar pattern. Since an image $I$ has a discrete number of pixels, it cannot be modeled as a continuous gradient space, so the derivative must be estimated by taking the smallest possible step $\Delta x = 1$ pixel:

$\frac{\partial I}{\partial x} \approx I(x+1,y) - I(x,y)$

The laplacian is the divergence of the gradient, measuring how rapidly the gradients in the space change. This conforms to the definition of the second derivative, which is found through the difference between the differences:

$\frac{\partial^2 I}{\partial x^2} \approx (I(x+2,y) - I(x+1,y)) - (I(x+1,y) - I(x,y))$

$\frac{\partial^2 I}{\partial x^2} \approx I(x+2,y) - 2I(x+1,y) + I(x,y)$

This results in Laplacian kernels that compute differences in local patches. However, this computed forward difference means the center of the filters are skewed to the right, shifting differences to the right and accumulating differences in the edges. A symmetric Laplacian kernel is preferred because it would accumulate differences in the center of each patch:

$\frac{\partial^2 I}{\partial x^2} \approx (I(x+1,y) - I(x, y)) - (I(x,y) - I(x-1,y))$

$\frac{\partial^2 I}{\partial x^2} \approx I(x+1,y) - 2I(x, y) + I(x-1,y)$

The entire Laplacian is across both the x and y directions (width/height) of the image:

$\nabla^2 I = \frac{\partial^2 I}{\partial^2 x^2} + \frac{\partial^2 I}{\partial^2 y^2}$

$\nabla^2 I \approx I(x+1,y) - 2I(x, y) + I(x-1,y) + I(x,y+1) - 2I(x, y) + I(x,y-1)$

$\nabla^2 I \approx I(x+1,y) - 4I(x, y) + I(x-1,y) + I(x,y+1) + I(x,y-1)$

Since each location on an image can be represented as a vector, the Laplacian can be written as a linear combination by exposing the coefficients:

$\nabla^2 I \approx (1) \cdot I(x+1,y) - (4) \cdot I(x, y) + (1) \cdot I(x-1,y) + (1) \cdot I(x,y+1) + (1) \cdot I(x,y-1)$

Since the Lacplacian operation is element wise, each coefficient scales the coordinate of the image, thus we can construct the following kernel matrix by mapping positions:

$$
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

The mean Laplacian score of an image with $N$ total pixels can be defined as $\mu_{\Delta}$:

$\mu_{\Delta} = \frac{1}{N} \sum_{i=1}^N \nabla^2 I_i$

The Laplacian variance is then:

$\sigma^2 = \frac{1}{N} \sum_{k=1}^N \left(\nabla^2 I_k - \mu_{\Delta} \right)^2$

### Resolution

Resolution determines the physical scale of a singular pixel. Higher resolution does not always mean better image quality, thus this acts more as a normalization layer for other metrics. A 48-megapixel (MP) image from a flagship spreads a leaf over 48 million of pixels. Even if the photo is slightly blurry and the transitions take up a larger spatial size, that size could be relatively smaller than one of a sharper photo from a 12 MP image from a budget smartphone, inflating the Laplacian variance score. In simpler words, the pixel patch to entire image ratio of the higher resolution photo can be higher than the pixel patch to entire image ratio of a lower resolution photo due to the sheer image size (4 times more) relative to a comparatively incremental spatial size increase. 

For this reason, a Normalized Sharpness Index $S_n$ normalizes the score based on the total pixel size $N$:

$S_n = \frac{\sigma^2}{\sqrt{N}}$

### Graininess

Budget phones don't have grainier phones just because the camera is worse, but also because its processor is less efficient. Digital images are fundamentally digitzed signals. Incoming photons in the analog domain strike the camera sensor, exciting electrons in precise locations that map to physical reality. These flowing electrons are an electrical current that an Analog-to-Digital (ADC) translates into discrete pixels. Cheap processors have to give up more excess thermal energy to perform tasks, which transfers to the electrons in the camera sensor's scilicon substrate, increasing their energy level in arbitrary locations independent of actual photon absorption, resulting in the electrical noise you see in photos.

Because this noise can create complicated pixel patterns in an image that are insignificant in the context of the photo (like the leaf lesion), they are areas of high variance. This can make using Laplacian variance quite catastrophic because there are these smooth transitions between small gradients that can quickly accumulate and inflate the score. This is why the Laplacian variance is computed after a highly smoothed version of the image is subtracted from the original, giving us a grain metric that is calculated on the pure fine noise residual.

The blurred image is produced through a convolution with a Guassian kernel that has entries sampled out of a choosen Guassian distribution. 


## Information Preservation Channels

Older flagships and modern budget phones can have a lot of traits in common. Older phones with a small physical sensor sizes can clip bright highlights to pure white pixels (255) and crush details to pure black (0) in shadows. For this reason, flagships included multi-frame High Dynamic Range (HDR) processing, which takes several photos at different exposure values. Short exposures can capture the textures of bright highlights without oversaturating them and medium or long exposures can capture shadows and fine details within darker regions by oversaturing the entire image. HDR imaging computes a weighted fusion of multiple exposure frames after alignment, and may be implemented either as a batch optimization or as  sequential recursive approximation depending on hardware constraints. Older flagships also commonly included exposure bracketing, tone mapping, and computational denoising techniques. However, the final output is still a reconstruction from multiple noisy observations rather than a direct measurement of the scene. In low-light or high-contrast conditions, the signal-to-noise ratio (SNR) can be limited, which affects how reliably fine detail can be recovered across exposures. This could result in artifacts coming from motion between frames, imperfect temporal alignment, and a trade-off between denoising and detail preservation. This motivates treating image quality as a measure of how much information is preserved through the full computational imaging pipeline.

### Highlight & Shadow Clipping

RGB color channels do not account for luminance, so a luminance-focused color space like CIELAB, in which the L channel respresents pure brightness or luminance independent of color, is used. Let $L(x,y) \in [0, 100]$ be the luminance value of a pixel. An exposure histogram $H$ can be constructed, which counts the frequency of each brightness level across all N pixels:

$H(b) = \sum_{j=1}^N {I}(L_k = b) \ \text{for} \ b \in [0, 100]$

where $I$ is the indicator function that equals 1 if the pixel's brightness $L_k$ matches the bin value $b$ and is 0 otherwise. N is the total number of pixels in the entire image. For any one specific brightness level like $b = 100$, the exposure histogram $H$ outputs the total pixel count for that brightness. As a whole, the histrogram reveals the count of all the unique brightnesses present across each pixel. 

The reliability governor utilizes an under exposure ratio $R_{\text{shadow}}$ computed from this histogram $H$. It counts the number of pixels $N_{\text{shadow}}$ that are clipped close to the extreme 0, range [0, 2]:

$N_{\text{shadow}} = \sum_{i=0}^1 H(i)$

The ratio of the number of clipped pixels to the total number of pixels gives us a metric for how under exposed/saturated the image is:

$R_{\text{shadow}} = \frac{N_{\text{shadow}}}{N}$

An over exposure ratio $R_{\text{highlight}}$ is computed similarly, counting the number of pixels $N_{\text{highlight}}$ that are clipped close to the extreme 100, range [98, 100]:

$N_{\text{highlight}} = \sum_{i=99}^{100} H(i), \quad \quad R_{\text{highlight}} = \frac{N_{\text{highlight}}}{N}$

These metrics quantify the proportion of pixels that lose recoverable detail due to saturation at either intensity extreme. Luminance variance is also computed:

```python
def exposure_metrics(img):
    N = img[0].numel()
    L = rgb2lab_l(img)
    H = torch.histc(L, bins=100, min=0.0, max=100.0)
    
    N_s = H[0:2].sum()
    R_s = (N_s / N).item()

    N_h = H[98:100].sum()
    R_h = (N_h / N).item()
    
    Lvar = torch.std(L).item()
    
    return {"shadow clipping": R_s, "highlight clipping": R_h, "luminance variance": Lvar}
```

### Signal-to-Noise Ratio (SNR)
The SNR is defined as the ratio of signal power to noise power, this metric estimates the amount of useful visual information contained in an image relative to its overall intensity variation. Images with a high SNR generally contain cleaner distinguishable leaf structures, while low-SNR images tend to show increased sensor noise or poor imaging conditions that can reduce diagnostic reliability.

Assuming the luminance image consists of a signal component and noise component: $I(x, y) = S(x, y) + N(x, y)$
We approximate the signal power by the mean squared luminance:

$P_{\text{signal}} = \frac{1}{N} \sum_{i=1}^N x^2_i$
where $x_i$ denotes the luminance of the i-th pixel.
The noise power is approximated by the luminance variance:
$P_{\text{noise}} = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2$
where the mean image luminance $\bar{x}$ is computed as:

$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$

The estimated SNR can thus be computed as:

$\text{SNR} = \frac{P_{\text{signal}}}{P_{\text{noise}}}$

To make the data more concrete, we estimate in decibels (dB) to convert the linear scale to a logarithmic one, representing the relative strength of this signal to its noise:

$\text{SNR} = 10\log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$

Within CitraScan, this metric is not intended to measure the true physical sensor SNR. Instead it helps qauntify whether the captured leaf image contains sufficient visual information for reliable disease classification.

### Foreground Complexity - Edge Density
It's important that this metric is purely mathematical and model independent, as we want to ensure the reliability metrics don't depend on model performance. When leaf coverage is low, it means the image contains more foreground or other non-essential elements to the classification, possibly reducing diagnostic accuracy and confidence. Since it is difficult to measure leaf coverage purly mathematically, and equally challenging to do the same for foreground elements, we propose an Edge Density score that utilizes an edge detector called Canny. The edge map $E(x, y)$ uses Canny to detect edges in the grayscale luminance image I(x,y):

$$E(x, y) = \begin{cases} 1 & \text{if an edge is detected}\\ 
0 &  \text{otherwise} \end{cases}$$

Camera sensors can inherently introduce noise. Because edge detection relies on finding sharp transitions between pixels, noise can mimic a tiny edge. Thus, Canny utilizes an algorithm that applies a convolutional Gaussian filter to smooth out the image. Next, another convolutional filter, the Sobel, computes a horizontal and vertical gradient for each pixel patch that the kernel slides over. Note that it does not compute the second derivative like the Lapclacian, it computes the first derivative:

(a) Horizontal gradient 
$G_x \approx I(x+1,y) - I(x,y)$

We use the central difference instead of forward/backward because we want the differences to accumulate in the center, we value symmetry:

$G_x \approx I(x-1,y) - I(x+1,y)$

Realize this can be written as a linear combination:

$$G_x = (1)I(x-1,y) + (0)I(x,y) + (-1)I(x+1, y)$$

$$G_x = I \cdot \begin{bmatrix} 1 & 0 & -1\end{bmatrix}$$

(b) Vertical gradient

$$G_y \approx I(x,y-1) - I(x,y+1)$$

$$G_y \approx (1)I(x,y-1) + (0)I(x, y) + (-1)I(x,y+1)$$

$$G_y = I \cdot \begin{bmatrix} 1\\
0\\
-1\end{bmatrix}$$

A binomial approximation of a Guassian filter is used for smoothing:

$$\begin{bmatrix}
1\\
2\\
1\\
\end{bmatrix}$$

To find vertical edges, we want to minimize noise vertically, so the smoothing is applied vertically. Vertical edges are edges that vary horizontally, as they appear as a uniform boundary vertically. Thus, this kernel must jointly compute the smoothing vertically and central difference horizontally, so the Sobel filter is computed from a outer product between the smoothing vector and difference vector.:

$$\begin{bmatrix}
1\\
2\\
1\\
\end{bmatrix} \cdot \begin{bmatrix} 1 & 0 & -1\end{bmatrix} =  
\begin{bmatrix}
1 & 0 & -1\\
2 & 0 & -2\\
1 & 0 & -1
\end{bmatrix}$$

Likewise, a horizontal edge kernel will be computed from the outer product of a horizontal smoothing vector and a vertical central difference vector:

$$\begin{bmatrix}
1 & 2 & 1
\end{bmatrix} \cdot \begin{bmatrix} 1\\
0\\
-1\end{bmatrix} =  
\begin{bmatrix}
1 & 2 & 1\\
0 & 0 & 0\\
-1 & -2 & -1
\end{bmatrix}$$

Once these filters have both convolved independently with the image, there is a feature map of horizontal edges and a feature map of vertical edges. Remember these edges are essentially gradients of the smoothed out pixels. These feature maps are then combined into one by computing the L2 norm of the gradient, which is adding each squared pixel value and taking the square root of them:

$E(x, y) = G(x,y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$

Now we can turn this into a binary edge mask with a certain threshold. This allows for the calculation of the edge density metric:

$F_{\text{edge}} = \frac{1}{HW} \sum_{x=1}^H \sum_{y=1}^W E(x,y)$

where $H$ and $W$ are the image height and width, so $HW$ is the resolution of the grayscale luminance image. This produces a score between 0 and 1. Low edge density suggests that the image has a smooth background, single dominant object, and little visual clutter. High edge density suggests there may be many overlapping leaves, branches, weeds, textured background, resulting in potentially more difficult segmention. 

For normalization, we need to consider the maximum posisble values. So we would consider all the pixels in patch aligning with the positive values in the kernel having maximum luminance (1) while all the pixels aligning with the negative values in the kernel having minimum luminance (0). This results in a sum of 1 + 2 + 1 = 4. Thus, we normalize each kernel by 4.

### Bilateral Asymmetry Feature (BAF) Module
Standard Vision Transformers and CNNs may struggle with citrus diagnostics because they can develop reliance on color or texture-based shortcuts by comparing these features in localized spatial regions without broader context. As a result, diseases such as Zinc or Manganese deficiency can be frequently misclassified as HLB due to shared leaf yellowing patterns. 

Before convolutional layers take the masked inputs from the segmentor, feature tensors $F \in \mathbb{R}^{C \times H \times W}$ are first reflected horizontally to compute bilateral differences. The difference between the flipped feature map and original feature map quantifies how different the opposite lateral sides of the feature maps are, so a higher difference indicates asymmetric features while a lower difference indicates symmetric features, constructing a feature map only containing the asymmetric features:

$$\Delta A = |F - \text{Flip}_{H}(F)|$$

where $\text{Flip}_{H}$ reflects features horizontally across the width axis ($W$). 

The resulting Asymmetry Map ($\Delta A$) is concatenated directly with the original feature tensor:

$$F_{\text{augmented}} = \text{Concat}([F, \Delta A], \text{dim}=1)$$
