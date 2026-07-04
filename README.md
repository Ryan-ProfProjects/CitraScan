# CitraScan-Pipeline
A hardware-agnostic computer vision pipeline designed to democratize early-warning detection of Huanglongbing (HLB) for farmers across varying field conditions.

Imagine your entire livelihood is on the line because a deadly disease is destroying your orange grove. You pull out your smartphone to use a disease-detection app, but the AI fails because the image is blurry, the lighting is poor, or the leaf isn’t perfectly framed. Today’s agricultural AI is often trained on clean ideal images that don’t reflect the messy conditions of real farms. At CitraScan, we are providing open-source data so that AI models can learn to perform reliably in the field. Our mission is to ensure cutting-edge diagnostics belong to the farmers who need it most, regardless of the smartphone in their pocket.

The CitraScan pipeline evaluates incoming data with an unsupervised algorithmic data fairness governor. Multiple metrics are leveraged to account for blurriness, resolution, lighting conditions, and symptom positioning in a structured Fairness Vector. 

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

Because this noise can create complicated pixel patterns in an image that are insignificant in the context of the photo (like the leaf lesion), they are areas of high variance. This can make using Laplacian variance quite catastrophic because there are these smooth transitions between small gradients that can quickly accumulate and inflate the score. This is why the Laplacian variance is computed after a highly smoothed version of the image is subtracted from the original, giving us a grain metric that is calculated on the pure fine noise.

The blurred image is produced through a convolution with a Guassian kernel that has entries sampled out of a choosen Guassian distribution. 

### Governing Lighting Conditions

Low-end budget sensors have a narrow dynamic range. This means that in bright sunlight, highlights are clipped to pure white pixels (255), and in shadows, details are crushed into pure black (0). When an image is clipped like this, the transitions are abrupt/sharp and local pixels become more uniform. This increased unfiromity results in almost no difference between coordinates, causing certain localized gradients to vanish and lower the Laplacian variance score. High-end flagship sensors use multi-frame High Dynamic Range (HDR) processing, which takes several photos at different exposure values. Short exposures can capture the textures of bright highlights without oversaturating them and medium or long exposures can capture shadows and fine details within darker regions by oversaturing the entire image. These RAW frames are not stacked directly due to temporal differences. First, the medium-exposure image is chosen as the reference frame and a predetermined filter slides over pixel patches and calculates a mathematical similarity score with each frame across the temporal dimension, selecting which parts of that frame should overlap with the reference frame. This results in a highly engineered output image that contains relatively sharp shadows and highlights with minimal noise. Now the transitions between pixels become much more continuous, increasing the Laplacian variance. 

To eliminate a possible performance delta between device tiers and prevent the diagnostic model(s) from overfitting to hardware noise, the Fairness Governor implements an unsupervised exposure index EI. This quantifies how severely localizing lighting conditions and hardware limitations corrupted the incoming signal. If the model is trained mainly on pristine images with smooth gradients, it struggles to generalize to lower-quality images taken on budget devices because it hasn't learned to extract patterns with abrupt transitions or excess noise. Conversely, when the model does encounter more low-qaulity images, it risks overfitting to hardware artifacts. A neural network with high capacity can easily learn the highly-specific noise pattern instead of the actual geometric structures of plant lesions. The EI combines over/under exposure ratio and luminance entropy.

RGB color channels do not account for luminance, so a luminance-focused color space like LAB, in which the L channel respresents pure brightness or luminance independent of color, is used. Let $L(x,y) \in [0, 255]$ be the luminance value of a pixel. An exposure histogram $H$ can be constructed, which counts the frequency of each brightness level across all N pixels:

$H(b) = \sum_{j=1}^N {I}(L_k = b) \ \text{for} \ b \in [0, 255]$

where $I$ is the indicator function that equals 1 if the pixel's brightness $L_k$ matches the bin value $b$ and is 0 otherwise. N is the total number of pixels in the entire image. For any one specific brightness level like $b = 255$, the exposure histogram $H$ outputs the total pixel count for that brightness. As a whole, the histrogram reveals the count of all the unique brightnesses present across each pixel. 

The fairness governor utilizes an over/under exposure ratio $R_{\text{clip}}$ computed from this histogram $H$. It counts the number of pixels $N_{\text{clipped}}$ that are clipped to each extreme(0 and 255):

$N_{\text{clipped}} = H(0) + H(255)$

The ratio of the number of clipped pixels to the total number of pixels gives us a metric for how over or under exposed/saturated the image is:

$R_{\text{clip}} = \frac{N_{\text{clipped}}}{N}$

The fairness governor also computes a luminance entropy $E$ from the histogram. This is derived from the shannon entropy definition of how much information that can be recieved from this lighting distribution. Due to vanishing gradients and abrupt transitions, poorly lit images have low luminance entropy while a well-lit image with increased detail and smooth transitions between gradients has high luminance entropy:

$E = -\sum_{b=0}^{255} p(b)log_2p(b)$

where $p(b) = \frac{H(b)}{N}$ is the probability of a pixel having brightness $b$. 

Since a higher over/under exposure ratio (or clipping ratio) means the signal is more clipped, the opposite ratio $1- R_{\text{clip}}$ is how un-clipped the signal is. Thus, $1- R_{\text{clip}}$ is porportional to the exposure index. Higher luminance entropy means the ligting carries more detail, so it is also directly porportional to the exposure index. However, if the brightness levels are uniformly distributed (each brightness level shows up the same number of times as the others in the image), the probability $p(b)$ of any brightness level $b$ would be like rolling a fair dice with 256 possible states, $\frac{1}{256}$. Mathematically, this makes the luminance entropy $E = 8$, so the maximum exposure index with the largest clipping ratio of $1.0$ would be $8$. To normalize it, a constant $0.125$ is included, and thus the exposure index EI is defined as:

$EI = \frac{1}{8}(1-R_{\text{clip}})E$

Raw EI scores are min-max scaled so that relative differences are exposed in a clean distribution:
```python
# min-max scaling to exaggerate relative differences
eis = (eis - eis.min()) / (eis.max() - eis.min())
eis = np.clip(eis, 0.0, 1.0)
```

A fairness vector is constructed by the column-wise stacking of each metric, and there exists three quality thresholds for each metric. The qaulity thresholds were computed empirically from the flagship images and simulated images:

```python
metrics = ["Blurriness", "Graininess", "Exposure Index (EI)"]
for i, name in enumerate(metrics):
    print(name)
    print(f"Good images (min / max): {good_imgs[:, i].min()} / {good_imgs[:, i].max()}")
    print(f"Bad images (min / max): {bad_imgs[:, i].min()} / {bad_imgs[:, i].max()}")
    # mid-point threshold
    print(f"threshold: {(good_imgs[:, i].min() + bad_imgs[:, i].max()) / 2}")
```
