---
title: 'Course Review GNR 638'
date: 2024-06-23
permalink: /posts/2024/06/course-review-gnr-638/
tags:
  - Deep Learning
  - Computer Vision
  - IIT Bombay
---

## GNR 638 - Deep Learning for Computer Vision

**Year:** 2023-24 Spring Semester  
**Instructor:** Prof. Biplab Banerjee

### Motivation

The rapid advancements in deep learning have revolutionized the field of computer vision, enabling machines to interpret and understand visual data with unprecedented accuracy. By mastering these techniques, you will be equipped to solve complex visual problems, drive innovation in various industries, and push the boundaries of what machines can achieve in visual perception.

### Course Content

The official course content can be found [here](https://www.csre.iitb.ac.in/mtechProgramme.php). The updated syllabus is listed below:

- Feature extraction of images, Festure descriptor
- Perceptron, PTA, Sigmoid neuron, MLP as universal function approximator, Depth vs width in MLP, Backpropagation in MLP, One to one mapping between the loss function and the probability model of prediction.
- Weight initialisation (Xavier-Glorot and He), Internal covariance shift, Normalisation (Batch, Layer, Instance, Group), Dropout and dropconnect regularization, Learning activation functions, EMA, Gradient descent with momentum, RMS prop, AdaGrad, Adam optimizer
- Convolution operation and arithmetic, Pooling, Dialated convolution, Separable convolution, 1 by 1 convolution, Deconvolution, Modern CNN architectures.
- PCA vs AutoEncoder, Denoising AE, Sparse AE and Contractive AE, Variational inference, ELBO, Loss function of VAE, Representation trick, Forward KL divergence vs Reverse KL divergence, Disentanglement by Beta VAE.
- Discriminative vs generative models, MLE principle, MLE with incomplete data, MAP estimate of Gaussian RV, Transformation of Gaussian RV, Fourier transform of Gaussian RV, Gaussian mixure model, MLE of GMM, EM algorithm in GMM, Conjugate prior, Exponential family of RV, Bayesian estimation, Kernel density estimation, Implicit and explit generative model.
- GAN framework as minimax game, Discriminator and generator cost function, Optimal discriminator of GAN, Optimal generator of GAN and JS divergence, Mode collapse and vanishing gradient in GAN, Earth mover distance, WGAN loss function as primal-dual linear program, WGAN-GP, Info-GAN.
- RNN, Backpropagation through time, Vanishing and exploding gradients, Capturing long term dependencies using LSTM and GRU.

### Feedback on Lectures

- Teaching Style: While the instructor demonstrated excellent communication skills, the course’s fast-paced nature often assumed a significant amount of prior knowledge on ML and probability theory, which could be challenging for some students. Despite these challenges, the instructor excelled in providing valuable resources that greatly supplemented the learning experience. However, the slides could benefit from better organization and increased depth to align with the course requirements, particularly for the mathematically rigorous exams. For students willing to engage deeply with the provided materials and seek additional help as needed, this course offers a comprehensive and rewarding exploration of deep learning techniques in computer vision.
- Attendence: Taken.

### Feedback on Assignments and Exams

- Weightage: Assignments - 0% (Negative if not submitted), Quizzes - 10%, Blog post - 10%, Midsem - 20%, Two Mini projects - 30%, Endsem - 30%
- Pattern: The mini projects, while time-consuming, provided valuable hands-on experience, though the evaluation based entirely on metric scores was seen as less than ideal by some students. The mid-semester and end-semester exams were notably difficult and highly mathematical, requiring a solid understanding of the theoretical concepts. The blog post assignment was a more balanced component, providing a good opportunity for reflection and synthesis. Overall, the evaluation was strict, but it ensured a deep and thorough understanding of the material for those who persevered.

### Difficulty Level

The course strikes a balance between accessibility and challenge. While the mini projects are manageable and offer practical insights, the exams are notably more demanding, requiring a solid grasp of mathematical concepts. To excel in these assessments, a significant amount of self-study is essential. Overall, the course is of moderate difficulty, providing a comprehensive and rewarding learning experience for those willing to invest the necessary time and effort.

### Prerequisites

While the course has no formal prerequisites, it is designed with the expectation that students are well-versed in the basics of machine learning. A solid understanding of neural networks, including CNN and RNN architectures, is assumed. Although probability topics are crucial for the coursework, they will not be covered in the lectures, requiring students to have prior knowledge in this area. On a positive note, the instructor does not assume any prior experience with generative models such as VAEs and GANs, making this aspect of the course more accessible to all students.

### Grading Stats

| Grade | Count |
|-------|-------|
| AA    | 29    |
| AB    | 36    |
| AP    | 1     |
| AU    | 4     |
| BB    | 18    |
| BC    | 33    |
| CC    | 27    |
| CD    | 15    |
| DD    | 9     |
| FR    | 1     |
| **Total** | **173** |

### Reference Books

- [Deep Learning - Stanford](https://cs231n.stanford.edu/schedule.html)
- [Deep Learning - CMU](https://deeplearning.cs.cmu.edu/F24/index.html)
- [GAN Paper](https://arxiv.org/abs/1406.2661)
- [VAE Paper](https://arxiv.org/abs/1312.6114)
- [Probability - Random Variables and Stochastic Processes, Athanasios Papoulis, S Pillai](https://www.youtube.com/channel/UC3l1RPdC7259bQZ8JWQYdrw)
- [Dive Into Deep Learning](https://d2l.ai/)
- [Deep Learning, Bishop](https://www.bishopbook.com/)
- [Deep Learning, Ian Goodfellow](https://www.deeplearningbook.org/)

### Reviewed by

Soumen Mondal (Email: [23m2157@iitb.ac.in](mailto:23m2157@iitb.ac.in))
