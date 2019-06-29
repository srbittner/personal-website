---
title: "Normalizing flows"
published: true
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

This week, I gave a mini-tutorial on normalizing flows after attending the 2019 ICML workshop on invertible neural networks and normalizing flows ([link](https://invertibleworkshop.github.io/){:target="_blank"}).  A few people around the theory center asked for my slides, so I figured I'd write a post covering the key ideas.

## Normalizing flows 101 ##

Normalizing flows model complex distributions as a transformation of a simple base density by an invertible function.  This base density is usually an isotropic gaussian, as is common in deep generative modeling, or maybe a uniform or other distribution.  Normalizing flows are often described as a sequence of invertible, differentiable transformations $$f : \mathcal{R}^D \rightarrow \mathcal{R}^D$$ of such a base density: 

$$u \sim q_0(u)$$

$$z_K = f_K \circ ... \circ f_2 \circ f_1(u)$$

Why is the invertibility of each $$f$$ so critical?  The key property of normalizing flows is that they facilitate an efficient calculation of sample density.  Unlike ordinary deep generative models like variational autoencoder (VAE) decoders and generative adversarial networks (GANs), normalizing flows provide samples _and_ sample density in one forward pass of the network.  This sample density calculation relies on the change of variables formula, which is only valid for invertible transforms $$f$$.  Specifically, the density of a variable $$z' = f(z)$$ is

<center><h3>Change of variables formula</h3></center>

$$q(z') = q(f^{-1}(z')) \left| \det \frac{\partial f^{-1}(z')}{\partial z'} \right| = q(z) \left| \det \frac{\partial f(z)}{\partial z} \right|^{-1}$$

Normalizing flows are thusly named because they _normalize_ the probability density of their samples as they step through each stage of the generative process (the _flow_).  The middle expression is the evaluation of the density as a function of $$z'$$ and the right expression is evaluated as a function of $$z$$ the variable at the previous stage of the normalizing flow.  One expression may be more computationally tractable than the other, so it is wise to consider which is faster, when the variables at each stage of the normalizing flow are available.  The key is that these determinant jacobian (or log determinant jacobian) calculations are fast in normalizing flows making optimization tractable.  Fully-connected bijective neural network layers have cubic time (very slow) log determinant jacobians, so they are not considered to be normalizing flows despite their invertibility.  Normalizing flows research is focused on the  design of function families that are expressive, yet have tractable log determinant jacobians.

So, why exactly do we care about the density of samples from our generative model?  For one, normalizing flows allow us to do density estimation with deep generative models, which are usually noninvertible in practice.  Another application of normalizing flows I use in my research projects is the approximation of maximum entropy distributions ([16](#Loaiza-Ganem2017maximum)).  Perhaps most importantly, normalizing flows can be used as flexible approximations to posterior distributions when doing variational inference(VI). An increasingly popular strategy for training VAEs is to replace mean-field gaussian approximate posteriors with a more flexible normalziing flows model ([1](#Rezende2015variational)). 

![a](/images/06_26_19/VI_wNF.png){:width="80%"}

<center>(Rezende & Mohammed, 2015)</center>

In a VAE using normalizing flows as an approximate posterior model, the inference network (bottom left, a standard neural network) learns a mapping between data points, and the parameters of the normalizing flow (top left) a deep generative model of the posterior.  To get samples from this approximate posterior in the VAE, you would sample $$u \sim q_0$$, and pass it through the normalizing flow $$z = f_K(f_{K-1}(..f_1(u)))$$.

## Overview ##
Below, I've posted my diagram of the landscape of machine learning research on normalizing flows.  I'm not going to describe all of these methods in detail, just the arrows between groups of approaches that signify key insights that led to important transitions in the field. 

<center><h3>Normalizing flows research</h3></center>
![a](/images/06_26_19/NFs.png){:width="100%"}


## Crafty bijections ##
Early research on normalizing flows concerned the design of bijective $$\mathcal{R}^D \rightarrow \mathcal{R}^D$$ transforms that were expressive, yet had a fast log determinant jacobian calculation (LDJ).  Two examples of such crafty bijections are planar and radial flows ([1](#Rezende2015variational)).  Planar flows place a hyperplane somewhere in $$\mathcal{R}^D$$ such that points are pulled or pushed away from this plane depending on their distance from the plane.  This can be thought of as a one-layer neural network.  Radial flows place a point in $$\mathcal{R}^D$$, and points are pulled or pushed away from this point according to their distance from it.  Both of these transormations are invertible, have fast LDJs, and can be relatively expressive when stacking several such bijections into a normalizing flow.  Below are examples of simple densities transformed by these classes of normalizing flows.

![a](/images/06_26_19/crafty_bijections.png){:width="80%"}

<center>(Rezende & Mohammed, 2015)</center>

A key insight that led to the next wave of research was the realization that autoregressive models with gaussian conditional distributions are actually normalizing flows.

## Autoregressive flows ##
### AR models ###
Autoregressive (AR) models such as MADE ([17](#Germain2015masked)), NADE ([18](#Uria2016neural)), PixelCNN ([19](#Salimans2017pixelcnn++)), PixelRNN ([20](#Oord2016pixel)), and Wavenet ([21](#Oord2016wavenet)) have been very successful.  In these AR models, the $$i^{th}$$ dimension of observation $$x$$ is only dependent on the 1 through $$(i-1)^{th}$$ dimensions of $$x$$.  AR models factorize the density of $$x$$ as a product of these conditional distributions, so that the density is calculable.

$$p(x) = \prod_{i=1}^D p(x_i \mid x_{1:i-1}) $$

AR style deep generative modeling is popular for density estimation, since most deep generative modeling approaches do not admit a calculable density.  At face value this seems like a pretty strong inductive bias, especially for images.  However, the use of deep learning to parameterize gaussian approximations to these AR conditional distributions has yielded state-of-the-art log-likelihoods on important image, video, and audio data sets.

$$p(x_i \mid x_{1:i-1}) = \mathcal{N}(x_i \mid \mu_i, (\exp \alpha_i)^2), \text{      } \mu_i = f_{\mu_i}(x_{1:i-1}), \text{      } \alpha_i = f_{\alpha_i}(x_{1:i-1}) $$

In these AR models, $$f_{\mu}$$ and $$f_{\alpha}$$ are ususally standard deep neural networks.

### IAFs and MAFs ###
In the paper introducing inverse autoregressive flows (IAFs, ([2]), Kingma et al. explain that all autoregressive flows with gaussian conditional distributions are actually normalizing flows.  This is by virtue of the fact that we can think of the AR model as an affine transformation of a standard gaussian sample $$u \sim \mathcal{N}(0, I)$$.  This model admits a lower-triangular Jacobian, allowing us to calculate the log determinant jacobian by taking a sum of the standard deviations of the gaussian conditional distributions.

$$x_i = u_i \exp \alpha_i + \mu_i $$

$$x = g(u)$$

$$\log \left| \det \frac{ \partial g(u)}{\partial u} \right| = \sum_{i=1}^D \log \alpha_i$$

In IAFs, the mean and variance of the gaussian conditional distributions is defined conditionally on the previous dimensions of the previous layer, whereas in masked autoregressive flows (MAFs, ([3](#Kingma2016masked))),  they are a function of the previous dimensions of the output of the transformation.  IAFs and MAFs are actually inverses of one another, and have a clear set of tradeoffs.  IAFs are fast to sample due to parallelizability, yet calculating the density of a sample requires iterating over each of the $$D$$ dimensions in the backward pass.  MAFs have fast, parallelizable density evaluation, but are slow to sample.

Fast sampling makes IAFs better suited than MAFs for approximating posteriors in variational inference, where we must sample frequently from our approximate posterior.  MAFs are better suited for density estimation, where we must frequently invert transformations in the normalizing flow to calculate the density.  

In many cases, it is unreasonable to assume such a strong inductive bias embodied by an AR model.  In practice, one "layer" of IAF or MAF is followed by a permutation layer, in which the indices of the data are randomly shuffled.  Then another IAF/MAF layer is applied.  This is done repetitively to reduce the level of inductive bias in this approach.

For some helpful diagrams on IAFs and MAFs, I highly recommend checking out Eric Jang's modern normalizing flows tutorial ([link](https://blog.evjang.com/2018/01/nf2.html){:target="_blank"}).

### Coupling layers ###

The next big step in normalizing flows research was the introduction of coupling transforms.  In models like NICE ([4](#Dinh2014nice)) and real NVP ([5](#Dinh2016density)), the latter $$D-d$$ dimensions of $$x$$ are conditioned on the first $$d$$ dimensions.  This makes the forward and backward passes (sampling and density evaluation, respectively) parallelizable, yet reduces the expressiveness of the model.  Real NVP uses coupling transforms and affine transformations between elements of subsequent layers as in IAF and MAF.  Like IAF and MAF, permutations are applied between layers to reduce index-ordering related inductive bias.  NICE is Real NVP with $$\alpha=0$$, so no scaling, just shifts.  In NICE, the volume of the distribution never changes throughout the sequence of shifts.

In the paper on Glow ([6](#Kingma2018glow)), permutation layers of Real NVP are replaced with invertible 1x1 convolutions.  Invertible matrices are constructed by a fixed permutation matrix, a lower triangular matrix of 1s, and learned upper triangular matrix and diagonal of positive elements. The result in sizable log likelihood gains.  Below of some examples of samples from a Glow model trained on a popular celebrity faces data set.

![a](/images/06_26_19/Glow.png){:width="80%"}
<center>(Kingma et al. 2018)</center>

## Affine to nonaffine ##

Most recent work on normalizing flows has been using either the autoregressive or coupling layer architecture with new types of invertible univariate functions to replace the affine transforms from layer to layer.  The hope is that more complicated univariate functions can increase the expressivity of the noramlizing flows at a small increased cost of calculating the jacobian of the univariate function.  Popular approaches have been to use neural networks ([7](#Huang2018neural), [8](#DeCao2019block)), splines ([9](#Durkan2019cubic), [10](#Durkan2019neural)), and several other approaches ([11](#Jaini2019sum), [12](#Ziegler2019latent), [13](#Ho2019flow++)).

Neural autoregressive flows (NAFs), are obviously autoregressive, and I believe that this choice is made because the universality proofs require such autoregressive structure.  However, there is nothing inhibiting one form using a coupling layer architecture and these monotonic neural networks to parameterize the transform.  It seems as though the paper on neural spline flows (NSFs, [10](#Durkan2019neural)) contains the most holistic comparison of all of these options, demonstrating general superiority of NSFs.

## Neural ordinary differential equations ##

Finally, a completely alternative approach to normalizing flows has evolved in parallel to these advancements in the papers on NODE  ([14](#Chen2018neural)) and FFJORD  ([15](#Grathwol2018ffjord)).  In NODE, a system of ordinary differential equations parameterizes the transformation of a base density to a more complex one.  By solving the ODEs backwards, we can obtain the density of these samples.  In FFJORD, the Hutchinson trace estimator is used to get efficient estimates of such densities.

Here are the [slides](/talks/normalizing_flows.pdf) from the tutorial.

## References ##
<a name="Rezende2015variational"></a>1. Rezende, Danilo Jimenez, and Shakir Mohamed. "*[Variational inference with normalizing flows](https://arxiv.org/abs/1505.05770){:target="_blank"}*." arXiv preprint arXiv:1505.05770 (2015).

<a name="Kingma2016improved"></a>2. Kingma, Durk P., et al. "*[Improved variational inference with inverse autoregressive flow](http://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow){:target="_blank"}*." Advances in neural information processing systems. 2016.

<a name="Papamakarios2017masked"></a>3. Papamakarios, George, Theo Pavlakou, and Iain Murray. "*[Masked autoregressive flow for density estimation](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation){:target="_blank"}*." Advances in Neural Information Processing Systems. 2017.

<a name="Dinh2014nice"></a>4. Dinh, Laurent, David Krueger, and Yoshua Bengio. "*[Nice: Non-linear independent components estimation](https://arxiv.org/abs/1410.8516)*." arXiv preprint arXiv:1410.8516 (2014).

<a name="Dinh2016density"></a>5. Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "*[Density estimation using real nvp](https://arxiv.org/abs/1605.08803){:target="_blank"}*." arXiv preprint arXiv:1605.08803 (2016).

<a name="Kingma2018glow"></a>6. Kingma, Durk P., and Prafulla Dhariwal. "*[Glow: Generative flow with invertible 1x1 convolutions](http://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions){:target="_blank"}*." Advances in Neural Information Processing Systems. 2018.

<a name="Huang2018neural"></a>7. Huang, Chin-Wei, et al. "*[Neural autoregressive flows](https://arxiv.org/abs/1804.00779){:target="_blank"}*." arXiv preprint arXiv:1804.00779 (2018).

<a name="DeCao2019block"></a>8. De Cao, Nicola, Ivan Titov, and Wilker Aziz. "*[Block neural autoregressive flow](https://arxiv.org/abs/1904.04676){:target="_blank"}*." arXiv preprint arXiv:1904.04676 (2019).

<a name="Durkan2019cubic"></a>9. Durkan, Conor, et al. "*[Cubic-Spline Flows](https://arxiv.org/abs/1906.02145){:target="_blank"}*." arXiv preprint arXiv:1906.02145 (2019).


<a name="Durkan2019neural"></a>10. Durkan, Conor, et al. "*[Neural Spline Flows](https://arxiv.org/abs/1906.04032){:target="_blank"}*." arXiv preprint arXiv:1906.04032 (2019).


<a name="Jaini2019sum"></a>11. Jaini, Priyank, Kira A. Selby, and Yaoliang Yu. "*[Sum-of-Squares Polynomial Flow](https://arxiv.org/abs/1905.02325){:target="_blank"}*." arXiv preprint arXiv:1905.02325 (2019).

<a name="Ziegler2019latent"></a>12. Ziegler, Zachary M., and Alexander M. Rush. "*[Latent Normalizing Flows for Discrete Sequences](https://arxiv.org/abs/1901.10548){:target="_blank"}*." arXiv preprint arXiv:1901.10548 (2019).

<a name="Ho2019flow++"></a>13. Ho, Jonathan, et al. "*[Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design](https://arxiv.org/abs/1902.00275){:target="_blank"}*." arXiv preprint arXiv:1902.00275 (2019).

<a name="Chen2018neural"></a>14. Chen, Tian Qi, et al. "*[Neural ordinary differential equations](http://papers.nips.cc/paper/7892-neural-ordinary-differential-equations){:target="_blank"}*." Advances in Neural Information Processing Systems. 2018.

<a name="Grathwol2018ffjord"></a>15. Grathwohl, Will, et al. "*[Ffjord: Free-form continuous dynamics for scalable reversible generative models](https://arxiv.org/abs/1810.01367){:target="_blank"}*." arXiv preprint arXiv:1810.01367 (2018).

<a name="Loaiza-Ganem2017maximum"></a>16. Loaiza-Ganem, Gabriel, Yuanjun Gao, and John P. Cunningham. "*[Maximum entropy flow networks](https://arxiv.org/abs/1701.03504){:target="_blank"}*." ICLR (2017).


<a name="Germain2015masked"></a>17. Germain, Mathieu, et al. "*[Made: Masked autoencoder for distribution estimation](http://proceedings.mlr.press/v37/germain15.pdf){:target="_blank"}*." International Conference on Machine Learning. 2015.

<a name="Uria2016neural"></a>18. Uria, Benigno, et al. "*[Neural autoregressive distribution estimation](http://www.jmlr.org/papers/volume17/16-272/16-272.pdf){:target="_blank"}*." The Journal of Machine Learning Research 17.1 (2016): 7184-7220.

<a name="Salimans2017pixelcnn++"></a>19. Salimans, Tim, et al. "*[Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications](https://arxiv.org/abs/1701.05517){:target="_blank"}*." arXiv preprint arXiv:1701.05517 (2017).

<a name="Oord2016pixel"></a>20. Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "*[Pixel recurrent neural networks](https://arxiv.org/abs/1601.06759){:target="_blank"}*." arXiv preprint arXiv:1601.06759 (2016).

<a name="Oord2016wavenet"></a>21. Van Den Oord, Aäron, et al. "*[WaveNet: A generative model for raw audio](https://regmedia.co.uk/2016/09/09/wavenet.pdf){:target="_blank"}*." SSW 125 (2016).

