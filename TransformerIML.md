# Discussion of the applicability existing Transformer IML methods to TabPFN

The techniques discussed in the following rely on the internal mechanisms of a Transformer model. These methods primarily focus on computing relevance scores for features, indicating their significance in the probability to fall into a certain class,
and can be classified as Feature Effect measures. Montavon et al. (2017) identified several desirable properties for relevance scores, including conservation, which ensures that the sum of relevance scores for all features equals the model’s
prediction.

## 1. Attention Weights
When considering the model class of Transformers, which includes TabPFN, a commonly used approach is to analyze the attention weights, which can be easily extracted. However, interpreting these weights as relevance scores
can often be misleading for two main reasons. Firstly, in deeper layers, the representation of a token is heavily influenced not only by the token itself but also by other tokens, thereby deviating from a straightforward
representation of the token input. Additionally, Transformers consist of network components beyond just attention layers, which are not taken into account when solely focusing on attention weights. While there are extensions
such as attention rollout that address some of these concerns, they still rely on assumptions that are not valid in practical scenarios (Bastings and Filippova, 2020, Chefer et al., 2021b, Böhle et al., 2023, Ali et al., 2022,
Abnar and Zuidema, 2020).

## 2. Functional Approaches
Additional approaches can be classified into two categories: functional approaches and message passing approaches (see further below). Functional approaches often rely on the theoretical basis of Taylor decomposition.
These methods involve performing a first-order Taylor expansion of the model function $f_j(x)$ (predicted probability for the $j^{th}$ class) around a reference point $\widetilde{x}$. The summed elements in the second summand are
taken as relevance, e.g. the relevance score for the $i^{th}$ feature is $R_{i}$. When the root point serves as a reference point (leading to $f_j(\widetilde{x}) = 0$) and the  second-order and higher-order terms $\varepsilon$
can be considered insignificant, the conservation principle is satisfied. However, it is important to note that the reference point should not deviate significantly from the actual point in order for the Taylor expansion to remain valid
(Montavon et al., 2017).

```math
 f_j(x) = f_j(\widetilde x) + \left(\frac{\partial f_j }{\partial x}\Big|_{x = \widetilde x}\right)^{\!\top} \!\! \cdot (x-\widetilde x) + \varepsilon= f_j(\widetilde x) + \sum_{i} \underbrace{ \frac{\partial f_j }{\partial x_i}\Big|_{x = \widetilde x} \!\!\cdot ( x_i-\widetilde x_i )}_{R_i} + \, \varepsilon $$
```

### 2.1 Sensitivity Analysis
Sensitivity Analysis is a widely used functional approach that leverages gradients computed in a backward pass to extract relevance scores. Unlike analyzing attention weights, this method takes into consideration the entire computation path, thereby accounting for all components of the model. 
Sensitivity Analysis can be considered a particular application of Taylor decomposition, where the model function is expanded around a point infinitesimally close to the actual input point, rather than a root point.
However, this expansion results in a significant absorption of relevance, leading to the violation of the conservation principle. While Sensitivity Analysis captures the sensitivity of model outputs to changes in input features,
it does not account for the extent to which the feature value might have already influenced the prediction. Moreover,  to be able to compare the magnitude of their gradients, all input features must undergo standardization
(Montavon et al., 2017, Bastings and Filippova, 2020).

### 2.2 Input x Gradient
An extension of Sensitivity Analysis is the Input x Gradient method. In this approach, the gradient with respect to an input variable is multiplied by the input variable itself. This modification enhances its suitability as a relevance score compared to plain gradients. It can be seen as a form of Taylor expansion, where the reference point is assumed to have a value of zero and is treated as a root point simultaneously. However, this assumption often does not hold in practical scenarios, resulting in the violation of the conservation principle (Bastings and Filippova, 2020, Simonyan et al., 2013). Furthermore, the Input x Gradient method is not applicable to TabPFN when dealing with label-encoded categorical features. Regardless of the actual gradient, multiplying gradients with input variables for the first level of categorical features leads to relevance scores of 0. The formula is similarly given by:

```math
R_i= \frac{\partial f_j\left(\mathbf{\begin{bmatrix}
\mathbf{X}_{\text{$train$}} \\
\mathbf{x}_{\text{$test$}} \\
\end{bmatrix} }\right)}{\partial \mathbf{x}_\text{$test,i$}} 
\cdot (\mathbf{x}_\text{$test,i$})
```

## 3. Message passing approaches
Message passing approaches, also known as Backpropagation Techniques, involve propagating relevance scores layer by layer back to the input variables. Unlike gradient-based techniques, these approaches employ customized rules to recursively redistribute relevance scores.

### 3.1. Layerwise Relevance Propagation
One popular method in this category is Layerwise Relevance Propagation (LRP).
However, LRP assumes the use of rectified linear units (ReLU) as the non-linearity and lacks propagation rules for skip connections, making it unsuitable for TabPFN. To address this limitation, Chefer et al. (2021) modified LRP specifically for Transformer models. Nevertheless, their method is restricted to Transformer encoders that utilize self-attention modules and cannot be applied to TabPFN, which also involves cross-attention computations (Bastings and Filippova, 2020, Chefer et al., 2021, Montavon et al., 2017, Chefer et al., 2021, Binder et al., 2016, Ali et al., 2022, Hollmann et al., 2023).

### References

Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics.

Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R. and Wolf, L. (2022). Xai for transformers: Better explanations through conservative propagation, International Conference on Machine Learning, PMLR, pp. 435–451.

Bastings, J., & Filippova, K. (2020). The elephant in the interpretability room: Why use attention as explanation when we have saliency methods?. In Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP (pp. 149-155).

Binder, A., Montavon, G., Lapuschkin, S., Müller, K.-R. and Samek, W. (2016). Layerwise relevance propagation for neural networks with local renormalization layers, Artificial Neural Networks and Machine Learning–ICANN 2016: 25th International Conference on Artificial Neural Networks, Barcelona, Spain, September 6-9, 2016, Proceedings, Part II 25, Springer, pp. 63–71.

Böhle, M., Fritz, M. and Schiele, B. (2023). Holistically explainable vision transformers, arXiv preprint arXiv:2301.08669.

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 782-791).

Montavon, G., Lapuschkin, S., Binder, A., Samek, W. and Müller, K.-R. (2017). Explaining nonlinear classification decisions with deep taylor decomposition, Pattern recognition 65: 211–222.

Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F., (2023). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. In The Eleventh International Conference on Learning Representations (ICLR).

Simonyan, K., Vedaldi, A., & Zisserman, A. (2014, April). Deep inside convolutional networks: visualising image classification models and saliency maps. In Proceedings of the International Conference on Learning Representations (ICLR).
