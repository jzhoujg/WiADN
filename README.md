## WiADN: Asymmetrical Dual-Task AttentionNetwork for WiFi Sensing

This repo is the code for paper "WiADN: Asymmetrical Dual-Task AttentionNetwork for WiFi Sensing"

1| Abstract 

In this paper, we propose a system called WiADN which aims to address the asymmetrical problems in the joint recognition of users’ locations and activities.
Our system is composed of two critical parts: an asymmetrical network architecture and an adaptive weight loss (AWL) module employed during the training phase.
The key insight of the proposed architecture is to mimic the skilled learners in similar situations, who often tackle easier problems first to enable them to solve
more challenging problems later on.


2| The Asymmetric Architecture

The architecture of attention based multi-task networks. 

![image](https://github.com/user-attachments/assets/bd9998a7-772b-4acd-b409-be893fa2509e)


The unit of Attention module. Mask generator is employed for transfering the prior knowledge and feature extractor is employed for learning respective
characteristics.

![image](https://github.com/user-attachments/assets/d3ba15c7-5e2e-4deb-8196-4233818a975d)


3 | Adaptive Loss Weight Based on Task Uncertainty 

Furthermore, an adaptive weight loss module based on task uncertainty is adopted to balance the tasks during the training phase.

4| Results


![image](https://github.com/user-attachments/assets/cc366435-6185-4070-b6b1-e276d6683db3)
