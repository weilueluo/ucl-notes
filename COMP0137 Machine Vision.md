# Machine Vision

> Book: http://www.computervisionmodels.com/

The big goal of machine vision is to:

- Recognition
  - Character / Face / Object / Action
- Reconstruction
  - 2D image $\rightarrow$ 3D model
- Tracking
  - Follow object in video
- Navigation
- Segmentation
- Enhancement Synthesis
- ...

Applications includes:

- Autonomous Vehicles
- Security
- Text Recognition
- Augmented Reality
- Image Retrieval
- Medical Image Analysis
- Model Building
- ...

History:

- 1970s: low level vision with binary images
- 1980s:  close with animal vision
- 1990s: estimation of camera pose and scene geometry
- 2000s: close with ML, CNN...

## Introduction

Machine Vision is hard because of (1) its dimensionality, all combinations of pixel values is a huge number (though we will never see most of them); (2) It is a inverse problem, i.e. the mapping from scene to image is many-to-one, it is not unique; (3) real-time is hard because video contains many data. Luckily, we know about how graphics works, have prior knowledge that expects what we will see in the image (help non-uniqueness), and there are huge data available.

## Overview

1. Probability

   - Joint probability

   - Conditional probability

   - Independence

   - Bayes’ rule

   - Common probability distributions

   - How to fit distributions to data

   - The multivariate normal

2. ML for vision: infer world from data

   - Model for regression (Discriminative)

     - Linear & non-linear regression
     - Gaussian process regression
     - The relevance vector machine

     - Models for classification (Discriminative)
       - Logistic regression
       - Gaussian process classification
       - Boosting and classification trees

   - Model complex PDFs (Generative)
     - EM algorithm
     - Mixture models
     - t-distributions
     - Factor Analysis

3. Connecting Models

   - Conditional Independence

   - Graphical Models

   - Inference on tree-structured models

   - Pictorial Structures

   - Undirected models

   - Markov random fields

   - Graph cuts

4. Models of Shape

   - Point distribution model

   - Active Shape Models

   - Active appearance models

5. Tracking

   - The Kalman Filter

   - Extensions of the Kalman Filter

   - Particle Filtering

6. Face Recognition

   - Subspace models for recognition

   - Within- and between- individual variance

   - Recognition across pose

7. Geometry of a Single Camera

   - Image transformations
   - How do 3d points project to pixels
   - Special cases of imaging

8. Geometry of multiple cameras

   - Stereo vision
   - Epipolar geometry
   - Finding and matching distinctive keypoints 
   - Shape from silhouette

## Probability

- **Random Variable** Output of a function that you do not know the input, the function can be discrete or continuous.

- **Probability** Probability of a output from a random variable.

- **Joint Probability** Probability of two random variable outputs happen together.
  $$
  \begin{split}
  Pr(x|y)&=\frac{Pr(x,y)}{Pr(y)}
  \end{split}
  $$

- **Bayes Rule**
  $$
  \begin{split}
  Pr(y|x)=\frac{Pr(x|y)Pr(y)}{Pr(x)}
  \end{split}
  $$
  $Pr(y)$ can be seem as remove the denominator in $Pr(x|y)$ to get $Pr(x,y)$ and add $Pr(x)$ to get $Pr(y|x)$.

- **Expectation**
  $$
  E[f(x)]=\int{f(x)Pr(x)}\; dx
  $$

  - Calculating the expectation requires us to know probability of each input.

  > TODO: what is moment

### Probability Distributions

> The explicit formula is not discussed, they are created by mathematician for us to model probabilistic distribution to have some mathematically convenient properties.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220423_1650668873.png" alt="image-20220423000750199" style="zoom: 80%;" />

### Bernoulli

> univariant, discrete, binary

$$
Pr(x)=Bern_x[\lambda]
$$

where
$$
\begin{align*}
Pr(1)&=\lambda\\
Pr(0)&=1-\lambda\\

\lambda&\in [0, 1]\\
x&\in\{0, 1\}

\end{align*}
$$
Or in one line:
$$
Pr(x)=\lambda^x(1-\lambda)^{1-x}
$$
- A probability distribution for two discrete values.

### Beta

> univariant, continuous, multivalues

$$
Pr(\lambda)=Beta_\lambda[\alpha, \beta]
$$

Beta distribution is used to represent a distribution of a value ranged between 0 and 1, called $\lambda$, controlled by $\alpha$ and $\beta$.

- The ratio of $\alpha$ and $\beta$ determine where the peak/expectation is.
- As the magnitude of $\alpha$ and $\beta$ increases, the steepness/concentration of the peak/expectation increase.
- This can be seen as an input for Bernoulli distribution.
- Expectation can be calculated as $\frac{\alpha}{\alpha+\beta}$.

![image-20220423111137986](https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220423_1650708699.png)

> TODO: usage

### Categorical

- multi-values version of Bernoulli distribution.
  - Each value $x_i$ follows $Pr(x=x_i)=\lambda_i$ and $Pr(x\neq x_i)=1-\lambda_i$.

<img src="COMP0137 Machine Vision.assets/image-20211011143108061.png" alt="image-20211011143108061" style="zoom: 33%;" />

### Dirichlet

- Continuous version of Categorical distribution.
- Multi-values version of Beta distribution.
  - Relative ratio of $a_{i\in K}$ determine the peak/expectation value.
  - Absolute value of $a_{i\in K}$ determine the height/concentration of the peak.

![image-20220423121156575](https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220423_1650712316.png)

### Univariate Normal

<img src="COMP0137 Machine Vision.assets/image-20211011154219447.png" alt="image-20211011154219447" style="zoom: 50%;" />

- $\mu$ describe the position of the peak.
- $\sigma$ describe the variance/width of the distribution.

### Normal Inverse Gamma

We model the distribution of the $\mu$ and $\sigma$ using 4 parameters: 

- $\alpha$: control the position of center, up or down.
- $\beta$: controls the spread of the center within the variance.
- $\gamma$: control the spread of variance.
- $\delta$: control the position of the center, left or right.

![image-20220423123439702](https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220423_1650713679.png)

### Multivariate Normal

- multi-value version of univariate normal
  - multiple positions, multiple variances, each control its own axes.

### Normal Inverse Wishart

- Suitable for describing uncertainty in the parameters of a multivariate normal distribution. Similar to Beta describing the parameter of Bernoulli; and normal inverse gamma describing parameters of univariate normal.
- It is just a function that produces a positive value for any valid mean vector µ and covariance matrix Σ, such that when we integrate over all possible values of µ and Σ, the answer is one.

![image-20220423132514253](https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220423_1650716714.png)

- α spread of covariance
- Ψ average covariance
- γ spread of mean
- δ average mean

### Conjugate

|                     | conjugate(*similar*) to |                             |
| ------------------- | ----------------------- | --------------------------- |
| Bernoulli           |                         | Beta                        |
| categorical         |                         | dirichlet                   |
| univariate normal   |                         | normal-scaled inverse gamma |
| multivariate normal |                         | normal inverse wishart      |

When we multiply a distribution with its conjugate, the result is proportional to a new distribution which has the same form as the conjugate. For example:
$$
\operatorname{Bern}_{x}[\lambda] \cdot \operatorname{Beta}_{\lambda}[\alpha, \beta]=\kappa(x, \alpha, \beta) \cdot \operatorname{Beta}_{\lambda}[\tilde{\alpha}, \tilde{\beta}]
$$
where κ is a scaling factor that is constant respect to our variables. Proof:
$$
\begin{aligned}
\operatorname{Bern}_{x}[\lambda] \cdot \operatorname{Beta}_{\lambda}[\alpha, \beta] &=\lambda^{x}(1-\lambda)^{1-x} \frac{\Gamma[\alpha+\beta]}{\Gamma[\alpha] \Gamma[\beta]} \lambda^{\alpha-1}(1-\lambda)^{\beta-1} \\
&=\frac{\Gamma[\alpha+\beta]}{\Gamma[\alpha] \Gamma[\beta]} \lambda^{x+\alpha-1}(1-\lambda)^{1-x+\beta-1} \\
&=\frac{\Gamma[\alpha+\beta]}{\Gamma[\alpha] \Gamma[\beta]} \frac{\Gamma[x+\alpha] \Gamma[1-x+\beta]}{\Gamma[x+\alpha+1-x+\beta]} \operatorname{Beta}_{\lambda}[x+\alpha, 1-x+\beta] \\
&=\kappa(x, \alpha, \beta) \cdot \operatorname{Beta}_{\lambda}[\tilde{\alpha}, \tilde{\beta}]
\end{aligned}
$$


## Fitting data to normal distribution

- The input data: $X$.
- The size of $X$: $I$.

### Maximum Likelihood

We estimate the parameters using:
$$
\begin{aligned}
\hat{\boldsymbol{\theta}} &=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\operatorname{Pr}\left(\mathbf{x}_{1 \ldots I} \boldsymbol{\theta}\right)\right] \\
&=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\prod_{i=1}^{I} \operatorname{Pr}\left(\mathbf{x}_{i} \mid \boldsymbol{\theta}\right)\right]
\end{aligned}
$$
This is because in order to achieve maximum, $\theta$ must be configured such that its distribution closely fits the actual data $x_i$, because $Pr(x|\theta)=Pr(x,\theta)/Pr(\theta)$, by assuming that the distribution of $\theta$ is uniform, we need to maximize the overlapping area of $\theta$ and $x$, this is achieved when $\theta$ and $x$ have the same shape, i.e. same distribution.

To do this, we calculate the mean and variance of the data, in the case of modeling using Gaussian distribution.
$$
\begin{align*}
\hat{\mu}&=\text{mean}(X)&=&\frac{\sum_{i=1}^I(x_i)}{I}\\
\hat{\sigma}^2&=\text{variance(X)}&=&\frac{\sum_{i=1}^{I}(x_i-\hat{\mu})^2}{I}
\end{align*}
$$

### Maximize a Posterior

Now, what if our prior is not uniform? i.e. we already had some previous data available that hints us about what will happen. the prior $\hat{\mu}$ and $\hat{\sigma}^2$, which allow us to configure out the probability of the new  $\hat{\mu}$ and $\hat{\sigma}^2$ given the old  $\hat{\mu}$ and $\hat{\sigma}^2$. So we do:

$$
\begin{aligned}
\hat{\boldsymbol{\theta}} &=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\operatorname{Pr}\left(\boldsymbol{\theta} \mid \mathrm{x}_{1 \ldots I}\right)\right] \\
&=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\frac{\operatorname{Pr}\left(\mathrm{x}_{1 \ldots I} \mid \theta\right) \operatorname{Pr}(\boldsymbol{\theta})}{\operatorname{Pr}\left(\mathrm{x}_{1 \ldots I}\right)}\right] \\
&=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\frac{\prod_{i=1}^{I} \operatorname{Pr}\left(\mathrm{x}_{i} \mid \boldsymbol{\theta}\right) \operatorname{Pr}(\boldsymbol{\theta})}{\operatorname{Pr}\left(\mathrm{x}_{1 \ldots I}\right)}\right]\\
&=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}{\prod_{i=1}^{I} \operatorname{Pr}\left(\mathrm{x}_{i} \mid \boldsymbol{\theta}\right) \operatorname{Pr}(\boldsymbol{\theta})}

\end{aligned}
$$


### Bayesian

Here instead of trying to predict a certain $\hat{\theta}$, we compute a probability distribution for all $\theta$ using bayesian rule:

![image-20211018172025224](COMP0137 Machine Vision.assets/image-20211018172025224.png)

when given a new data, we compute its probability using:

![image-20211018172050183](COMP0137 Machine Vision.assets/image-20211018172050183.png)

It is probability of new data given a configuration, multiply by the probability of that configuration.

The goal is to calculate the probability of each configuration:
$$
P(\theta \mid D)=\frac{\overbrace{P(D \mid \theta)}^{likelihood} \overbrace{P(\theta)}^{prior}}{P(D)}=\frac{P(D \mid \theta) P(\theta)}{\int_{\theta} P(D \mid \theta) P(\theta) d \theta}
$$

### Comparison

> https://towardsdatascience.com/mle-map-and-bayesian-inference-3407b2d6d4d9

ML maximize $P(x|\theta)$ and MAP maximize $P(\theta|x)$, they return a single value, thus point estimator. Bayesian inference calculates the full posterior distribution, thus it return a PDF.

- Other point estimator exists such as expect a posterior (EAP)

### Covariance

Here we introduce three type of covariances:
$$
\boldsymbol{\Sigma}_{\text {spher }}=\left[\begin{array}{cc}
\sigma^{2} & 0 \\
0 & \sigma^{2}
\end{array}\right] \quad \boldsymbol{\Sigma}_{\text {diag }}=\left[\begin{array}{cc}
\sigma_{1}^{2} & 0 \\
0 & \sigma_{2}^{2}
\end{array}\right] \quad \boldsymbol{\Sigma}_{\text {full }}=\left[\begin{array}{cc}
\sigma_{11}^{2} & \sigma_{12}^{2} \\
\sigma_{21}^{2} & \sigma_{22}^{2}
\end{array}\right]
$$

- Spherical: scaled identity.
- Diagonal: diagonal.
- Full: symmetric and positive definite.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220424_1650800412.png" alt="image-20220424124010323" style="zoom:80%;" />

### Properties

If we have normal distribution of variable $x$:
$$
Pr(x)=\operatorname{Norm}_x[\mathbf{\mu}, \mathbf{\Sigma}]
$$
Then we have variable $y=Ax+b$ then its corresponding normal distribution is:
$$
Pr(x)=\operatorname{Norm}_x[A\mu+b, A\Sigma A^T]
$$
So to draw a sample of any distribution by transforming it from the standard normal distribution (mean 0, covariance identity), we first draw $x$ from a normal distribution, then apply $y=\Sigma^{-1}x+\mu$.

> TODO: how is this dervied

- If we marginalize a multi-variate normal, then the result also follows normal distribution

- If we condition on a subset of multi-variate normal on the using the rest of the variables, the conditioned distribution is also normal.

- The product of two normal distribution is also another normal distribution:
  $$
  \begin{aligned}
  \operatorname{Norm}_{\mathbf{x}}[\mathbf{a}, \mathbf{A}] \operatorname{Norm}_{\mathbf{x}}[\mathbf{b}, \mathbf{B}] =
  \kappa \cdot \operatorname{Norm}_{\mathbf{x}} {\left[\left(\mathbf{A}^{-1}+\mathbf{B}^{-1}\right)^{-1}\left(\mathbf{A}^{-1} \mathbf{a}+\mathbf{B}^{-1} \mathbf{b}\right),\left(\mathbf{A}^{-1}+\mathbf{B}^{-1}\right)^{-1}\right] }
  \end{aligned}
  $$
  <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220424_1650818731.png" alt="image-20220424174529560" style="zoom:80%;" />

- If  we have normal distribution over a variable $x=Ay+b$, then we can express the same normal distribution in term of $y=A'x+b'$ as well. It is often used in bayesian rule to move $P(x|y)$ to  $P(y|x)$.

## Computer Vision

We take visual data $x$, infer world state $w$ (discrete/continuous), the measure data $x$ is often noisy and can map to many $w$, so best we can do is return a posterior distribution $P(w|x)$. We need **model**  to relate $x$ and $w$; **learning algorithm** that takes a pair of $(x_i, w_i)$ to fit parameters $\theta$; and an **inference algorithm** that takes the model and a new $x$ and return the probability distribution of the world $P(w|x,\theta)$, or draw sample from learned posterior.

- Inference is simpler with discriminative model, because we can directly compute $P(w|x)$; generative model make inference via bayesian rule which can be computational expensive.
- Generative model is built on $P(x|w)$ and discriminative model built on $P(w|x)$. Since the world space can be much smaller than the data, e.g. image space vs some aspect of the world, it may be more costly to build discriminative model.
  Secondly, the parameter used to describe the data maybe much larger than describing the world, even thought both configuration describing the data maybe referring to the same world state, and these redundancy can be expensive.
- The process of how data is created is more close to $P(x|w)$, we can account for perspective projection and occulusion, other approach requires learning these phenomena from the data.
- Generative model model the joint distribution over all data dimensions and can effectively interpolate missing elements.
- Generative model allows us to use expert prior knowledge as prior, which is harder for discriminative model.

Generative models are more common.

Applications include: skin detection and background subtraction.

#### Why multivariate model may not work

- It is unimodal, may not represent well by a single peak.
- not robust, single outlier can dramatically affects the estimate of the mean and covariance.
- too many parameters, the covariance matrix contains $D(D+1)/2$ parameters, sometimes forced to use diagonal form.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220424_1650833595.png" alt="image-20220424215313629" style="zoom:80%;" />

#### Hidden Variable

Just a unknown distribution, we will then have a joint probability and we will marginalize it out later:
$$
\operatorname{Pr}(\mathbf{x} \mid \boldsymbol{\theta})=\int \operatorname{Pr}(\mathbf{x}, \mathbf{h} \mid \boldsymbol{\theta}) d \mathbf{h}
$$
The model involved often have hidden variable, in these cases, it will have neat close form solution only if we consider the hidden variable (which we cannot as it is hidden). We now need to apply non-linear optimization techniques or the expectation maximization algorithm to the right hand side of below equation, similar to ML:
$$
\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}\left[\sum_{i=1}^{I} \log \left[\int \operatorname{Pr}\left(\mathbf{x}_{i}, \mathbf{h}_{i} \mid \boldsymbol{\theta}\right) d \mathbf{h}_{i}\right]\right]
$$

> continue: pg108
