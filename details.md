#Program Details
##Optimizer:Adam
We use Adam algorithm for our optimization.Adam is an adaptive learning rate optimization algorithm and is presented as follow.The name "Adam" derives from the phrase "adaptive moments".It can be seen as a variant on the combination of RMSProp and momentum with a few distinctions.Because of its efficiency, Adam has become one of the most popular optimization algorithm since  proposed in 2015.
<div align="center"> 
<img src="https://github.com/lsa1997/images/blob/master/adam.jpg?raw=true" width="80%"><br>
Adam algorithm</div>

In our program, we use the default settings ($\alpha=0.001,\beta_1=0.9,\beta_2=0.999$ and $\epsilon=10^{-8}$) with learning rate decay $0.95$.

To improve the efficiency, we changed the order of computation by replaceing the last three lines in the loop with following lines:$\alpha_t = \alpha\cdot\sqrt{1 - \beta^t_2}/(1-\beta^t_1)$ and $\theta_t\leftarrow \theta_{t-1}-\alpha_t\cdot m_t/(\sqrt{v_t}+\hat\epsilon)$

##Initialization:Xavier
It is important to initialize the deep neural network properly. In our program, we use the Xavier Initialization:
$$ W \sim U[-\frac{\sqrt 6}{\sqrt{n_j+n_{j+1}}},\frac{\sqrt 6}{\sqrt{n_j+n_{j+1}}}],$$
where $n_j$ and $n_{j+1}$ denote the input and output dimension for the j-th layer, respectively.
Xavier initialization, or the normalized initialization, aims to approximately maintain activation variances and back-propagated gradients variances as one moves up or down the network. It has been proved to perform well in practice.
##Accelerate convolution:im2col
The implement of naive convolution is very inefficient. Now most networks choose to use GPUs to accelerate this process.Unfortunately we don't have enough time to write our own CUDA codes, but we can use the method to change convolution into matrix product.A simple example is shown as follow.
<div align="center"> 
<img src="https://pic1.zhimg.com/v2-c4ebe9cee894b3294081ad503445c900_r.jpg" width="80%"><br>
Convolution using matrix</div>
We use some fancy index operation to implement this method.Though still pretty slow compared to GPU-based methods, it performs much better than the naive convolution.

##Batch Normalization
Expect the basic convolution layer ,pooling layer and fully-connected layer, we also used Batch Normalization to improve the perfomance of our network. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. 
The BN transform is shown as follows:
<div align="center"> 
<img src="https://github.com/lsa1997/images/blob/master/bn.jpg?raw=true" width="80%"><br>
Batch Normalizing transform</div>
The gradients with respect to the parameters of the BN transform are: 

$$ \frac{\partial l}{\partial\hat x_i}=\frac{\partial l}{\partial y_i}\cdot\gamma$$
$$ \frac{\partial l}{\partial\sigma^2_\mathcal{B}}=\sum_{i=1}^m\frac{\partial l}{\partial\hat x_i}\cdot(x_i-\mu_\mathcal{B})\cdot\frac{-1}{2}(\sigma^2_\mathcal{B}+\epsilon)^{-3/2}$$
$$\frac{\partial l}{\partial\mu_\mathcal{B}}=(\sum_{i=1}^m\frac{\partial l}{\partial\hat x_i}\cdot\frac{-1}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}})+\frac{\partial l}{\partial\sigma^2_\mathcal{B}}\cdot\frac{\sum_{i=1}^{m}-2(x_i-\mu_\mathcal{B})}{m} $$
$$\frac{\partial l}{\partial x_i}=\frac{\partial l}{\partial\hat x_i}\cdot\frac{1}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}}+\frac{\partial l}{\partial\sigma^2_\mathcal{B}}\cdot\frac{2(x_i-\mu_\mathcal{B})}{m}+\frac{\partial l}{\partial\mu_\mathcal{B}}\cdot\frac{1}{m} $$ 
$$ \frac{\partial l}{\partial\gamma}=\sum_{i=1}^m\frac{\partial l}{\partial y_i}\cdot\hat x_i$$ 
$$\frac{\partial l}{\partial\beta}=\sum_{i=1}^m\frac{\partial l}{\partial y_i}$$ 

During the training process, we calculated the running average of mean and variance with $momentum=0.9$, which are used for the testing process.

##Reference
1.Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014.

2.Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[C]//Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010: 249-256.

3.Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

