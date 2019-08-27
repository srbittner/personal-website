---
title: "Fast hessians of deep generative models"
published: true
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

Hessians are useful objects to have when analyzing system models.  For example, their eigendecomposition can be used to identify stiff and sloppy dimensions of model parameters.  For good resources on these ideas, check out [James Sethna's website](http://www.lassp.cornell.edu/sethna/Sloppy/WhatAreSloppyModels.html).

Deep generative models typically do not have calculable Hessians, because they generally do not have tractable densities.  DGMs with tractable densities called [normalizing flows](https://srbittner.github.io/2019/06/26/normalizing_flows/), admit tractable Hessians.  This blog post is a guide on how to compute them efficiently.

## Hessians of probability models ##
The Hessian of a probability model is the second order gradient of the model log density $$\log q_\theta(z)$$ with respect to the parameters $$z$$:

$$\frac{\partial^2 \log q_\theta(z)}{\partial z \partial z^\top}$$

For example, let $$q_\theta(z)$$ be an approximate posterior distribution.  At a given parameter choice $$z_0$$, we can examine the Hessian $$\frac{\partial^2 \log q_\theta(z_0)}{\partial z \partial z^\top}$$ to determine what dimensions of parameter space are critical for describing the data (high magnitude eigenvalue), and which are degenerate (low magnitude eigenvalue).  

So, for deep generative models $$q_\theta(z)$$, where we have a deterministic function $$z = f(\omega)$$, with random $$\omega \sim q_0$$, how do we calculate the Hessian?  Well, first we need to be able to calculate $$\log q_\theta(z)$$, which necessitates the use of a normalizing flow architecture.  With a normalizing flow, we have a function $$g(\omega; \theta) = \log q_\theta(z)$$.

We can't simply calculate $$\frac{\partial^2 g(\omega; \theta)}{\partial z \partial z^\top}$$, since $$g$$ is not a function of $$z$$.  Instead we must also calculate $$\frac{\partial \omega \partial \omega^\top}{\partial z \partial z^\top}$$ to obtain

$$\frac{\partial^2 \log q_\theta(z)}{\partial z \partial z^\top} = (\frac{\partial \omega \partial \omega^\top}{\partial z \partial z^\top})^\top\frac{\partial^2 g(\omega; \theta)}{\partial \omega \partial \omega^\top}$$


## You need the inverse of the sampler $$z = f(\omega)$$ ##
You might be wondering if the inverse function theorem is useful here.  The IFT states that for a function $$F$$ mapping point $$p$$ to $$q$$, $$q = F(p)$$, the jacobian of the inverse of F at $$q$$ is equal to the inverse of the Jacobian of F at $$p$$:

$$J_{F^{-1}}(q) = J_F(p)^{-1}$$

This may be obvious to some audiences, but you can't simply re-apply the IFT for second-order derivatives.  This means that in order to calculate the model Hessian, you must calculate intermediate Hessians with respect to $$z$$ of each element in the outer product of $$\omega \omega^\top$$.

## Amortization via element-wise Hessians ##
You could just run `tf.Hessian()` or the pytorch equivalent on your computational graph and call it a day, but you will have substantial repetition in the hessian graph, resulting in long build and exectute times.  We see this by writing out the functional form for the $$i,j^{th}$$ $$\omega = f^{-1}(z)$$ Hessians.  For simplicity, let's just call $$h(z) = f^{-1}(z)$$.

$$h : \mathcal{R}^D \rightarrow \mathcal{R}^D$$

$$\frac{\partial h(x)_i h(x)_j}{\partial x} = h(x)_i \frac{\partial h(x)_j}{\partial x} + \frac{\partial
h(x)_i}{\partial x} h(x)_j $$

The first gradient is simply the product rule for gradients.  The Hessian is then the re-application of the product rule for each term.

$$\frac{\partial^2 h(x)_i h(x)_j}{\partial x \partial x^\top} = h(x)_i \frac{\partial^2 h(x)_j}{\partial x \partial
x^\top} +  \frac{\partial h(x)_i}{\partial x} (\frac{\partial h(x)_j}{\partial x})^\top + \frac{\partial^2 h(x)_i}{\partial x
\partial x^\top}h(x)_j $$

We can simplify this expression into a dot product of the $$i^{th}$$ and $$j^{th}$$ functions values, first and second-order gradients.

$$\frac{\partial^2 h(x)_i h(x)_j}{\partial x \partial x^\top} =
\begin{bmatrix} h(x)_i & \frac{\partial h(x)_i}{\partial x} & \frac{\partial^2 h(x)_i}{\partial x \partial x^\top} \end{bmatrix}
\begin{bmatrix} \frac{\partial^2 h(x)_j}{\partial x \partial x^\top} \\ (\frac{\partial h(x)_j}{\partial x})^\top \\ h(x)_j \end{bmatrix}
$$

Since each element of the outer product is composed of such dot products, we can calculate the full $$\frac{\partial \omega \partial \omega^\top}{\partial z \partial z^\top}$$ using just the value, first-order and second-order gradients of each dimension of $$\omega$$ with respect to $$z$$.  By first calculating these for each dimension of $$\omega$$, we can reuse them in their necessary rows and columns of $$\frac{\partial \omega \partial \omega^\top}{\partial z \partial z^\top}$$.  In contrast, naively applying `tf.hessian()` would result in unnecessarily extra computation that increases as a square with dimensionality.


