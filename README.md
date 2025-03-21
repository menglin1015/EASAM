# EASAM
EASAM:An Edge-Aware SAM-Based Paradigm for Tooth Segmentation
# 
![图片](https://github.com/user-attachments/assets/263c131d-b80b-4400-94a8-6400433595bf)

# Abstract
Tooth segmentation in dental panoramic X-ray images is a task of great clinical importance. However, previous studies have often neglected the importance of edge information, resulting in inaccurate tooth segmentation with blurred borders and low contrast. In this paper, we propose EASAM, an edge-aware fusion transformer method designed to utilize edge information in order to assist salient features for dental panoramic X-ray image segmentation. A dual-stream structure is used, where the image coding layer of SAM is used for salient branching, and the other part uses a U-Net-like CNN structure for edge branching. Then the salient branch is fused with the edge branch, and the fused edge feature map is fused with the salient branch result to feed into the later structure. Extensive experiments on three public benchmark datasets demonstrate the effectiveness and superiority of our proposed method compared to other state-of-the-art methods. The method demonstrates the ability to accurately identify and analyze tooth structure, thus providing important information for dental diagnosis, treatment planning, and research.
# Contribution
- We propose a specialized dental segmentation network, EASAM, designed for segmenting teeth in X-ray, to achieve superior performance on complex anatomical structures and ambiguous lesions. The EASAM model, integrated with the dental imaging system, provides doctors with an automated tool for the rapid and accurate analysis of dental and periodontal tissue images. With the EASAM model’s ability to segment the dental region, it can significantly assist in medical decision-making. For instance, by locating the apical root position, it can determine whether the tooth impacts the mandibular canal. It can also identify regions of the tooth that are difficult to distinguish with the naked eye, aiding in the diagnosis of its effects on the oral cavity.
- We designed an edge branch and an edge-aware module to enhance the feature representation of the image encoder in SAM.
- Across three benchmark datasets, our approach exceeds the performance of current state-of-the-art algorithms, as evidenced by comparative analyses.
# Installation
Following Segment Anything, python=3.8.16, pytorch=1.8.0, and torchvision=0.9.0 are used in SAMUS.
## Our code will be released soon.
