#set page(
  paper: "a4",
  margin: (top: 1.97in, bottom: 1.97in, left: 1.58in, right: 1.58in),
  numbering: none
)

#set text(font: "Times New Roman", size: 12pt)

#let tab_length = 0.5in
#let line_spacing = 0.65em
#set par(leading: 1.15 * line_spacing)

#align(center)[
  #text(size: 14pt, weight: "bold")[
    ADVERSARIAL ROBUSTNESS IN\
    MODEL COMPRESSION:\
    A COMPARATIVE STUDY OF\
    KNOWLEDGE DISTILLATION IN\
    SEMANTIC SEGMENTATION\
    ON BDD100K DATASET
  ]
]

#align(center + horizon)[#text(stretch: 75%, size: 16pt, weight: "bold")[Giovanni Dejan]]

#align(center)[#text(stretch: 75%, size: 14pt, weight: "bold")[]]

#align(center + bottom)[
  #text(stretch: 75%, size: 14pt, weight: "bold")[
    FACULTY OF COMPUTER SCIENCE AND INFORMATION TECHNOLOGY
    \
    UNIVERSITI MALAYA
    \
    KUALA LUMPUR
    \
    \
    \
    2026
  ]
]

#pagebreak()

#set page(
  numbering: "i",
  number-align: right,
  margin: (top: 2cm, bottom: 2cm, left: 4cm, right: 2cm),
)
#counter(page).update(2)

#align(center)[
  #text(weight: "bold")[
    UNIVERSITI MALAYA
    \
    ORIGINAL LITERARY WORK DECLARATION
  ]
]

#table(
  columns: (5fr, 4fr),
  stroke: 0pt,
  inset: 6.5pt,

  [Name of Candidate: Giovanni Dejan], [#strike(stroke: 1.5pt)[I.C.] / Passport No.: X3386742],
  table.cell(colspan: 2)[Matric. No: 24201613],
  table.cell(colspan: 2)[Name of Degree: Master in Artificial Intelligence],
  table.cell(colspan: 2)[Title of Dissertation (“this Work”): \ Adversarial Robustness in Model Compression: A Comparative Study of Knowledge Distillation in Semantic Segmentation on BDD100K Dataset],
  table.cell(colspan: 2)[Field of Study: \ Machine Learning],
)

#let signature_inset = 5pt
#box(
  stroke: 0pt + black,
  inset: (left: 15pt, rest: 6.5pt)
)[
  #set par(justify: true)
  #set enum(numbering: "(1)")
  I do solemnly and sincerely declare that:
  + I am the sole author/writer of this Work;
  + This Work is original;
  + Any use of any work in which copyright exists was done by way of fair dealing and for permitted purposes and any excerpt or extract from, or reference to or reproduction of any copyright work has been disclosed expressly and sufficiently and the title of the Work and its authorship have been acknowledged in this Work;
  +	I do not have any actual knowledge nor do I ought reasonably to know that the making of this work constitutes an infringement of any copyright work;
  +	I hereby assign all and every rights in the copyright to this Work to the Universiti Malaya ("UM"), who henceforth shall be owner of the copyright in this Work and that any reproduction or use in any form or by any means whatsoever is prohibited without the written consent of UM having been first had and obtained;
  +	I am fully aware that if in the course of making this Work I have infringed any copyright whether intentionally or otherwise, I may be subject to legal action or any other action as may be determined by UM.
]

#box(
  stroke: 0pt,
  inset: (left: 30pt, rest: 6.5pt)
)[
  #table(
    columns: (2fr, 1fr),
    stroke: 0pt,
    inset: (top: signature_inset, bottom: 15pt),
    [Candidate's Signature], [Date],
  )
]

Subscribed and solemnly declared before,

#box(
  stroke: 0pt + black,
  inset: (left: 30pt, rest: 6.5pt)
)[
  #table(
    columns: (2fr, 1fr),
    stroke: 0pt,
    inset: (top: signature_inset, bottom: signature_inset),
    [Witness' Signature], [Date],
  )
]

Name:
\
Designation:

#pagebreak()

#align(center)[
  #text(weight: "bold")[
    ADVERSARIAL ROBUSTNESS IN MODEL COMPRESSION:\
    A COMPARATIVE STUDY OF KNOWLEDGE DISTILLATION\
    IN SEMANTIC SEGMENTATION ON BDD100K DATASET
  ]
]
#v(1em)
#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[
      [WIP] ABSTRACT
    ]
  ]
]

#set par(first-line-indent: (amount: tab_length, all: true), leading: 2 * line_spacing, spacing: 2 * line_spacing, justify: true)

#v(2 * line_spacing)

Over the last decade, more and more vehicles are equipped with driving assistance technologies, such as cruise control and autonomous driving technologies. These technologies rely on semantic segmentation model deployed in the car's embedded system, allowing the drivers to lighten up their burden, as well as knowledge distillation that allows a compressed model to have a comparable performance with larger model. However, the lack of studies in adversarial robustness on distilled semantic segmentation models hinder the adoption, as adversarial attack poses a serious threat on system's performance and safety. Therefore, this study proposes Seg-ARD, which combines semantic segmentation with adversarial training, allowing model to be compressed and robust at the same time. Incorporating existing techniques from previous papers, Seg-ARD will be developed with _Adversarial Robust Distillation (ARD)_ as the baseline. As part of this study, ResNet-18 and MobileNetV2 will be used as encoders in the teacher and student models, respectively. The distilled model will be evaluated using segmentation intersection-based evaluation metrics, such as _Intersection over Union (IoU)_ and Dice coefficient, as well as robustness-based evaluation metric such as _Attack Success Rate (ASR)_. In addition, inference latency will also be evaluation as part of the model evaluation for real-time performance.

*Keywords: driving assistance technologies, embedded system, semantic segmentation, knowledge distillation, adversarial robustness.*

#pagebreak()

#align(center)[
  #text(weight: "bold")[
    ADVERSARIAL ROBUSTNESS IN MODEL COMPRESSION:\
    A COMPARATIVE STUDY OF KNOWLEDGE DISTILLATION\
    IN SEMANTIC SEGMENTATION ON BDD100K DATASET
  ]
]
#v(1em)
#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[
      [WIP] ABSTRAK
    ]
  ]
]

#v(2 * line_spacing)

#set text(lang:"my")
Sepanjang sedekad yang lalu, semakin banyak kenderaan dilengkapi dengan teknologi bantuan memandu, seperti kawalan pelayaran dan pemanduan autonomi. Teknologi ini bergantung pada model segmentasi semantik yang terpasang pada sistem terbenam kenderaan bagi meringankan beban mereka, serta penyulingan pengetahuan yang membolehkan model termampat mencapai prestasi yang setara dengan model yang lebih besar. Walau bagaimanapun, kekurangan kajian mengenai ketahanan terhadap serangan adversari pada model segmentasi semantik yang disuling menghambat penerimagunaannya, memandangkan serangan adversari mendatangkan ancaman yang serius terhadap prestasi dan keselamatan sistem. Oleh sebab itu, kajian ini mencadangkan Seg-ARD, yang menggabungkan segmentasi semantik dengan latihan adversari, sekaligus membolehkan model dimampatkan dan berdaya tahan pada masa yang sama. Dengan menerapkan teknik sedia ada daripada kajian terdahulu, Seg-ARD dibangun menggunakan _Adversarial Robust Distillation (ARD)_ sebagai penanda aras. Sebagai sebahagian daripada kajian ini, ResNet-18 dan MobileNetV2 masing-masing akan digunakan sebagai pengekod dalam model guru dan model pelajar. Model yang disuling akan dinilai menggunakan metrik berasaskan pertindihan segmentasi, seperti _Intersection over Union (IoU)_ dan pekali Dice, serta penilaian ketahanan seperti _Attack Success Rate (ASR)_. Selain itu, pendaman inferens turut dinilai sebagai sebahagiaan daripada prestasi masa nyata.

*Kata kunci: teknologi bantuan memandu, sistem terbenam, segmentasi semantik, penyulingan pengetahuan, ketahanan terhadap serangan musuh.*

#set text(lang: "en")

#pagebreak()

#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[
      ACKNOWLEDGEMENTS
    ]
  ]
]
#v(2 * line_spacing)

I am deeply grateful for the supervision of Dr. Lim Chee Kau during the writing of the proposal. He helped me with the literature review and scoping down the research problems.

#pagebreak()

// TABLE OF CONTENTS PAGE

// 1. Custom Heading Numbering: "CHAPTER 1:" for level 1, "1.1" for sub-levels
#set heading(numbering: (..nums) => {
  let pos = nums.pos()
  if pos.len() == 1 {
    text(size: 12pt, "CHAPTER " + str(pos.at(0)) + ":")
  } else {
    text(size: 12pt, pos.map(str).join("."))
  }
})

// 2. Format TOC Entries: Bold ONLY numbered level-1 Chapters
#show outline.entry.where(level: 1): it => {
  if it.element.func() == heading {
    if it.element.numbering != none {
      v(0.8em, weak: true)
      strong(it)
    } else {
      it
    }
  } else {
    it
  }
}

#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[
      TABLE OF CONTENTS
    ]
  ]
]
\

// Set indentation step for nested sub-headings
#set outline(indent: 1.8em)

#outline(
  title: none,
  indent: 1.8em,
)
#pagebreak()

#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[LIST OF FIGURES]
  ]
]
\
#outline(
  title: none,
  target: figure.where(kind: image)
)
#pagebreak()

#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[LIST OF TABLES]
  ]
]
\
#outline(
  title: none,
  target: figure.where(kind: table)
)

#pagebreak()

// ------------------- MAIN CONTENT (CHAPTERS) -------------------

#set page(numbering: "1")
#counter(page).update(1)

// Configure figure & table numbering to follow Chapter number (e.g. Figure 1.1, Table 1.1)
#set figure(numbering: (..nums) => {
  let chap = counter(heading.where(level: 1)).get().at(0, default: 1)
  str(chap) + "." + str(nums.pos().first())
})

#show heading: set block(below: 2 * line_spacing)
#set text(hyphenate: false)
#set par(first-line-indent: (amount: tab_length, all: true), leading: 2 * line_spacing, spacing: 2 * line_spacing, justify: true)
#show enum: set block(spacing: 2 * line_spacing)
#show figure: set block(breakable: true)

#align(center)[
  #heading(level: 1)[
    #text(size: 12pt)[INTRODUCTION]
  ]
]
#v(2 * line_spacing)

#heading(level: 2)[
  #text(size: 12pt)[Research Background]
]
#v(2 * line_spacing)

Over the last decade, the rise of machine learning capabilities and embedded technology has open a lot of opportunities, particularly in the use of self-driving cars and cruise control technology (Han et al., 2022). Cruise control and autonomous driving technologies help to reduce the cognitive load of the drivers, acting like a junior reliable copilot. Not only this reduces stress during road trip (Stapel et al., 2018), but cruise control has been proven to reduce leg strain (Motorist, 2024) and reducing fuel use by intelligently adjust the power required (Moawad et al., 2024; Park et al., 2013).

One of the techniques in machine learning that enable computer-assisted driving is semantic segementation. Semantic segmentation works through assigning each pixel in the image with a specific class (Lakshmanan et al., 2021; Long et al., 2015). In the context of driving assistance technologies, segmentation allows vehicles to have visual perception, which enable path planning (Rossolini et al., 2023; Saeedizadeh et al., 2024). Semantic segmentation has come a long way since its birth, with modern usages in computer-assisted driving (Yao et al., 2023).

The applications of semantic segmentation for automotive use cases are enabled by the clever engineering of machine learning in resource-constrained devices, in particular knowledge distillation. Knowledge distillation allows smaller models to achieve equivalent performance of the larger models through distillation from larger models. Smaller models are able to achieve comparable performance, whilst having lower inference time due to smaller size (Lê et al., 2026; Moslemi et al., 2024). Lower inference time is a must for real-time use cases, as latency may cause harm due to the lack of outputs for some frames.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Problem Statement]
]
#v(2 * line_spacing)

However, the rise of reliance of deep learning also meant that a new surface of attack has emerged. One dangerous threat to deep learning-based applications is called adversarial attack. Adversarial attack exploits a major weakness of machine learning models simply by applying perturbation in the input (Hall et al., 2023; Yerlikaya & Bahtiyar, 2026). Due to the neural network architecture, a small change in the input model can drastically alter the output. In the case of image classification task, it can lead to misclassification (Khamaiseh et al., 2022). And in the case of semantic segmentation, it can lead to invalidate segmentation, such as failure to detect road lanes (Arnab et al., 2020).

Although adversarial attacks have received substantial attention in image classification, they remain severly understudied in semantic segmentation. This could pose a danger to day-to-day use cases, as such vulnerability not only compromises real-world viability of segmentation models, but may also breach safety-critical regulations (Arnab et al., 2020). While various countermeasures have been proposed, but one prominent gap remains: the adversarial robustness of knowledge-distilled architectures for semantic segmentation tasks remain largely unexamined (Lê et al., 2026).

Therefore, to address this gap, this study proposes Seg-ARD, an adversarial robustness distillation framework designed for semantic segmentation that mitigates perturbation vulnerabilities while maintaining inference efficiency on resource-constrained platforms.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Research Questions]
]

#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  + How effective is the proposed adversarial distillation framework (Seg-ARD) in improving the adversarial robustness of compact student model compared to standard response-based model when subjected to multi-step PGD attacks?
  + To what extent does integrating adversarial training into knowledge distillation impact the clean segmentation fidelity across diverse driving conditions?
  + What is the impact of the proposed framework on inference latency and hardware utilization in resource-constrained deployment settings?
]

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Research Objectives]
]

#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  + To develop and adapt an adversarial robustness distillation framework for semantic segmentation using ResNet-18-based segmentation teacher model and MobileNetV2-based segmentation student model.
  + To evaluate and benchmark the trade-off between clean segmentation performance and adversarial robustness using BDD100K dataset.
  + To quantify and analyze the inference latency and efficiency of the robust distilled model to validate its feasibility for real-time autonomous driving applications.
]

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Research Scope and Contribution]
]
#v(2 * line_spacing)

This research will cover the development and experiments on distilled model. As part for this research, 10-step PGD attack will be used to test the model's resistance against each type of the attack.

What is outside of the scope is the integration on the actual automotive system. Such integration should be conducted by car manufacturers in a highly-regulated environment, as improper integration may result in the failure of the car's internal systems.

The contribution of this research to the industry and society is the knowledge gained from adversarial training will be useful to increase the model's safety, especially for automotive manufacturers who embed self-driving and/or cruise control systems in their lineups.

#pagebreak()
#align(center)[
  #heading(level: 1)[
    #text(size: 12pt)[LITERATURE REVIEW]
  ]
]
#v(2 * line_spacing)

This chapter will be divided into 3 sections. The first section is called "Image Segmentation", discussing about the theory of image segmentation and recent studies in the context of autonomous driving.

The second section is called "Knowledge Distillation," discussing about the theory, performance, and use cases of knowledge distillation, specifically in computer vision applications.

The third section is called "Adversarial Attack and Robustness," discussing about adversarial attack in machine learning applications and the training process to enhance model's resistance against the attack.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Image Segmentation]
]
#v(2 * line_spacing)

Image segmentation is a computer vision task to assign each pixel to a label according to the masking provided in the dataset (Lakshmanan et al., 2021; Minaee et al., 2021). It allows the extraction of meaningful information from images and videos by partitioning each frame into segments (Bachani et al., 2024). Image segmentation has been used in many applications, ranging from medical imaging (Abdoli et al., 2025), robotics (Corke, 2023), and autonomous driving systems (Rossolini et al., 2023; Yao et al., 2023). In this study, the main focus is on semantic segmentation based on BDD100K dataset.

This part literature review will be split into 3 parts: Semantic Segmentation, Instance Segmentation, and Real-world Applications.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Semantic Segmentation]
]
#v(2 * line_spacing)

Semantic segmentation is a segmentation model that involves classifying individual pixels in an image based on a semantic meaning. The goal is to categorize each pixel and divide it into meaningful segments. This process is essential for industries like medical imaging, where pinpointing specific structures is important for diagnosis, and for autonomous driving, where road marking is important (Rayed et al., 2024). Originally, semantic segmentation is done using classical approaches, such as thresholding and edge detection. However, recent advancements in deep learning has allowed researchers to design and develop semantic segmentation model based on neural network (Brar et al., 2024).

#figure(
  image("u-net.png", width: 80%),
  caption: [U-Net architecture (Ronneberger et al., 2015).]
)
#v(2 * line_spacing)

One of the most influencial architecture in the semantic segmentation field is the U-Net architecture. U-Net is originally developed by Ronneberger et al. (2015) and intended for semantic segmentation for medical imaging, specifically biomedical microscopic image segmentation. U-Net consists of two parts: the contraction path and the expansive path. The contracting path follows the typical architecture of a convolutional network, allowing the model to capture the context from the input. The expansive path consists of an upsampling of the feature map followed by a convolutional layer in order to produce the segmentation. Due to its promising performance, U-Net has been adapted for other use cases, such as road extraction (Z. Zhang et al., 2018) and 3D images (Çiçek et al., 2016).

#figure(
  image("deconvolutional-semantic-segmentation.png"),
  caption: [Deconvolution Network (Noh et al., 2015).]
)
#v(2 * line_spacing)

Apart from U-Net, a similar architecture called Encoder-Decoder has been developed. One of such architecture is called Deconvolution Network. Similar to U-Net, Deconvolutional Network consists of two parts: the encoder and the decoder. The encoder corresponds to feature extractor that transforms the input image to multidimensional feature representation, whereas the decoder acts as a segment shape generator, producing the segmentation mask that acts as probability map (Minaee et al., 2021; Noh et al., 2015). However, one major difference from U-Net is that there is no copy and crop mechanism. This is because Deconvolution Network does not use skip connection, which is present in U-Net (Noh et al., 2015).

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Instance Segmentation]
]
#v(2 * line_spacing)

Instance segmentation is the advanced version of semantic segmentation. This group of technique involves labeling pixels with classes and instances. This is different from semantic segmentation because semantic segmentation does not distinguish between instances (Gu et al., 2022). There are various techniques that have been studied and developed over the years, such as two-stage instance segmentation methods, multi-stage methods, and others.

Two-stage methods are based on the principle that instance segmentation process depends on the object detection and segmentation. Existing methods can be categorized into top-down segementation and bottom-up segmentation. However, researchers deemed two-stage methods are not enough, hence the research on multi-stage methods.

Multi-stage methods are modifications of two-stage methods. Multi-stage methods arise from the fact that two-stage methods did not fully utilize the relationship between object detection and segmentation tasks, therefore an improvement is deemed necessary (Gu et al., 2022). Within multi-stage methods, one such method is the cascade architecture, where there is an additional step of mask generation, between instance discrimination and object classification (Dai et al., 2016).

Instance segmentation seems promising for cruise control and self-driving. However, Gu et al. (2022) noticed that one of the main downsides of instance segmentation is the latency time. When Gu et al. (2022) measured the inference time of the instance segmentation models, most of the models have inference speed lower than 30 frames per second, rendering the model almost unusable in real time settings.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Real-world Applications]
]
#v(2 * line_spacing)

Yao et al., (2023) attempted to mix visual inputs from camera with a separate radar system. The study stated that the fusion of these 2 systems are widely used in _Advance Driving Assistance Systems (ADAS)_, including _Adaptive Cruise Control (ACC)_, _Lane Change Assist (LCA)_, and _Automatic Emergency Braking (AEB)_. The fusion system is able to operate throughout the day with minimal disturbance, even during worsening weather and lightning conditions. This study found out that radar does not necessarily increase the accuracy and reliability of the segmentation model. However, the analysis on adversarial attack is not included.

Another study conducted by Rossolini et al. (2023) included analysis of adversarial attacks in their study, showing that both targeted and untargeted  attacks easily fool semantic segmentation models. They attributed this weakness into the standard pixelwise cross-entropy (CE) loss function, hence the reason why a novel loss function is required specifically for semantic segmentation tasks. The outcomes of their study are the novel loss function, as well as a high-performant adversarial patch detection, which is able to overcome the limitation of patching-based adversarial attack methods, according to the experiments. However, this study lacked the discussion on distilled models.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Knowledge Distillation]
]
#v(2 * line_spacing)

_Knowledge Distillation (KD)_ is a training technique that enables a small model (a student) to learn from larger model (a teacher) (Lê et al., 2026). Originally developed by Buciluǎ et al. (2006), it gained popularity when Hinton et al. (2015) implemented KD for neural network-style architecture. The idea behind KD is to leverage the knowledge (including the generalization knowledge) the large teacher has learned, and then transfer it to the student during the training process. Knowledge distillation has been used in various applications, ranging from visual question-answering (Fang & Yang, 2023), medical imaging (Li et al., 2022), and unmanned aerial vehicles (Kargin & Petrenko, 2023).

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Response-based Distillation]
]
#v(2 * line_spacing)

#figure(
  image("response-based-kd.png"),
  caption: [Response-based knowledge distillation (Chariton, 2023).]
)
#v(2 * line_spacing)

Response-based distillation is the basic form of distillation. As the name implies, this technique is used when a student model learns to mimic the soft output probabilities of the teacher's model, which is called "soft targets." These soft targets are a bit different from the final answer, which is usually the answer with the highest probability, due to the nuanced view of the data, encoding crucial class relationships for generalization. This approach helps the student model understand the teacher's decision making process, improving performance and avoiding overfitting (Song et al., 2025).

Part of the response-based distillation is a formula called _Kullback-Leibler divergence_ (KL divergence). KL divergence is defined as a distance between an approximation of probability distribution $Q$ with a true probability distribution $P$. KL divergence allows the student model to mimic the probability distribution of the teacher, hence the increased capability of the student (Hinton et al., 2015). Mathematically, KL divergence is defined as this:
$
  D_(\K\L)(P || Q) = sum_(x in X) P(x) (P(x)) / (Q(x))
  \
  P(x) = sigma((z_text("teacher")) / T)
  \
  Q(x) = sigma((z_text("student")) / T)
$

#set par(first-line-indent: (amount: 0cm))

where $P(x)$ and $Q(x)$ can be defined as the logits (post-softmax outputs) generate by the teacher model and student model, respectively. $z_text("teacher")$ and $z_text("student")$ are logits (pre-softmax outputs) generated by teacher model and student model respectively. $sigma$ is the softmax function. The temperature $T$ is used soften the logits and normally set to 1 or higher to soften the probability.

#set par(first-line-indent: (amount: tab_length))

The goal is to minimize the divergence between 2 probability distributions. In this type of KD, a student is able to mimic the teacher if the divergence is small enough.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Feature-based Distillation]
]
#v(2 * line_spacing)

#figure(
  image("feature-based-kd.png"),
  caption: [Feature-based knowledge distillation (Chariton, 2023).]
)
#v(2 * line_spacing)

Feature-based distillation is a type of knowledge distillation used to transfer the teacher's intermediate layers to the student's (Xiao & Tong, 2025). In theory, this enables the student to capture the teacher's internal structures and relationships encoded in the feature maps (Yang et al., 2023). However, Chariton (2023) found out that not all techniques are able to effectively transfer the internal representation. Not only that the similarity function needs to be considered, but the model architecture itself needs to be carefully studied. Due to reduced flexibility, therefore this feature-based distillation may not be suited for this study.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Relation-based Distillation]
]
#v(2 * line_spacing)

While response-based and feature-based distillation techniques focus on distilling knowledge from individual samples, relation-based distillation focuses on the relation of different data points or different features itself. The relational information can be encoded in the form of a graph with various edge weights (Yang et al., 2023). Even though this seems ideal for the use of semantic segmentation, however the main challenge is the computational complexity to analyze the relationships between data points and feature layers (Acharya et al., 2024). The additional complexity, even only in the training process, will introduce bottlenecks, which is not preferrable.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Real-world Applications]
]
#v(2 * line_spacing)

Zhao et al. (2025) developed a knowledge distillation framework for autonomous driving called _Risk-Aware Knowledge Distillation (RAKD)_. RAKD employs a teacher based on reinforcement learning to supervise a vision-based student. The teacher is trained using bird's eye vision and risk-aware reward shaping, as the environment is designed with adversarial scenarios, such as aggresive driving. However RAKD is not focused towards adversarial attack, since the main focus is on chaotic environment in the form of aggresive driving. Moreover, BDD100K dataset is produced through videos from dashboard camera, rather than bird's eye view of a car.

Another study related to knowledge distillation in autonomous driving is a study conducted by R. Yu et al. (2025). One difference compared to Zhao et al. (2025) is that R. Yu et al. (2025) uses End-To-End, a neural network architecture that combines all inputs from sensors to produce output, which is the trajectory of the vehicle. The inputs come from front-facing cameras, as well as sensors. This study claimed that DistillDrive, the KD-based End-to-End model that they developed, was able to achieve 50% of collision reduction, albeit unclear whether the environment consists of aggresive driving or not. However, R. Yu et al. (2025) did not put much emphasis on adversarial training. And secondly, R. Yu et al. (2025) focused on End-to-End system, rather than a specific task such as semantic segmentation task.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Adversarial Attack and Robustness]
]
#v(2 * line_spacing)

Yerlikaya and Bahtiyar (2026) defines adversarial attack as a type of attack caused by extreme data samples that may cause unexpected results. Similarly, Shang et al. (2025) defines adversarial attack as a specially crafted examples inputs that induce misclassification in deep learning models. The machine learning model may misjudge the malicious input, which in turn may exploit the system's flaws. One of the reasons is that the perceptron in the model may not work as expected in extreme conditions, for example harsh weather conditions in autonomous driving system. Adversarial attack is not only a theoretical concept, but has been studied in real world cases, for example fraud detection (Lunghi et al., 2023), image classification (Khamaiseh et al., 2022; Shang et al., 2025), image segmentation (Arnab et al., 2020), and autonomous driving (Qiao et al., 2026).

The following sections will discuss adversarial attack and defense techniques in detail.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Attacking Techniques]
]
#v(2 * line_spacing)

As briefly mentioned, the goal of adversarial attack is to alter the output of the machine learning model, whether the output is a classification or segmentation. This section will be focusing on the attacking techniques found in computer vision.

The most basic form of attacking technique is the evasion attack. Qiao et al. (2026) describes evasion attack as an attack in the form of perturbed input crafted by advesaries to misled the machine learning models to produce an incorrect output, without altering its internal parameters. This kind of attack exploit's the model's sensitivity to small, often invisible changes in the input data, resulting in manipulation of the output. The most popular attack in this category is _Fast Gradient Sign Method (FGSM)_. FGSM works by perturbing the input $x$ along the sign of the gradient of the training loss function $bold(J)(theta, x, y)$ with respect to the input:
$
  eta = epsilon.alt text("sign")(bold(J)(theta, x, y))
$

#set par(first-line-indent: (amount: 0cm))

where $eta$ is the perturbation and $epsilon.alt$ is the magnitude of the perturbation. Goodfellow et al. (2014) have shown that even perturbation with a magnitude of 0.007 already induces misclassification on GoogLeNet model.

#set par(first-line-indent: (amount: tab_length))

In the semantic segmentation task, C. Xie et al. (2017) developed an attacking method called _Dense Adversary Generation (DAG)_. The goal of DAG is to study the transferrability of adversarial examples between computer vision tasks, specifically between semantic segmentation and object detection. C. Xie et al. (2017) found out that adversarial perturbations can be transferred between different architectures and different datasets. C. Xie et al. (2017) indicated that the perturbation inputs for classification task can be applied to semantic segmentation task, and vice versa.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Defensive Measures]
]
#v(2 * line_spacing)

Defensive measures against adversarial attacks are defined as a group of techniques that help machine learning model increase resistance against perturbed inputs (Yerlikaya & Bahtiyar, 2026). Various techniques have been studied, ranging from pre-processing techniques that remove perturbations (for example, gaussian blur) to adversarial training that expose models to adversarial examples in order for the model to enhance the decision boundaries (H. Zhang et al., 2019).

One of the most popular adversarial training techniques is _Projected Gradient Descent (PGD)_. Muhammad and Bae (2022) described PGD as training the model based on multi-step perturbed inputs, where single step of perturbed input is done using FGSM. PGD allows models to shape smoother decision boundary, increasing the resistance against attack. However, one main downside is the computing power. According, Muhammad and Bae (2022), the number of iterations in the PGD acts as the multiplier of the training cost. For example, PGD-7 means 7 iterations of PGD, which means the training process require 7 times compute power compared to normal training.

One technique to lower the need of intensive computing power in PGD process is to recycle the information of the loss' gradient. One of the most popular techniques in this category is called _Free Adversarial Training (FreeAT)_, which is developed by Shafahi et al. (2019). Originally developed for regular training task, this method has been adapted for knowledge distillation training, which is named _Fast Adversarial Robust Distillation (Fast-ARD)_ by Goldblum et al. (2020).

In the context of knowledge distillation, one technique to integrate robustness to the distillation process is called _Adversarial Robust Distillation (ARD)_. ARD works by modifying the existing loss function from the distillation process, as described in the formula by Goldblum et al. (2020):
$
  min_(theta) bb(E)_((X, y) ~ D) [alpha t^2 underbrace(text("KL")(S_(theta)^(t)(X + delta_theta), T^(t)(X)), text("Adversarially Robust Distillation loss")) + (1 - alpha) underbrace(cal(l)(S_(theta)^(t)(X), y), text("classification loss"))]
$

#set par(first-line-indent: (amount: 0cm))
where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $S_(theta)$ is the student model with parameters $theta$.
  - $T$ is the teacher model.
  - $X$ is the input features.
  - $y$ is the input labels.
  - $D$ is the data generating distribution.
  - $alpha$ is the proportion in which the model should take into account ARD loss.
  - $t$ is the temperature.
  - $text("KL")$ is the KL divergence used for probability distribution.
  - $cal(l)$ is the loss function, which accepts the outputs from the student model and the input labels for loss calculation.
]

#set par(first-line-indent: (amount: tab_length))
#v(2 * line_spacing)

Although ARD is an important improvement, however the study of ARD is focused on  image classification task, not semantic segmentation. Developing a new defense mechanism based on ARD will need to reevaluate the loss function, as the output of semantic segmentation model is a segmentation mask, not a single class.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Summary]
]
#v(2 * line_spacing)

By looking at the existing studies, it is noticeable that there is a gap that has yet to be filled in the form of developing a distilled semantic segmentation model with acceptable level of robustness for the use of autonomous driving. Existing adversarial training studies have  yet to discuss about semantic segmentation, while semantic segmentation discussions do not touch the issue of robustness.

#pagebreak()
#align(center)[
  #heading(level: 1)[
    #text(size: 12pt)[RESEARCH METHODOLOGY]
  ]
]
#v(2 * line_spacing)

#heading(level: 2)[
  #text(size: 12pt)[Research Design]
]
#v(2 * line_spacing)

The research design will be a comparative study. However, due to the novel nature of this study, the comparison will be made between these student models:
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - A student distilled from a non-robust teacher.
  - A student distilled from a robust teacher.
  - A student distilled from a non-robust teacher using Segmented ARD (Seg-ARD) technique.
  - A student distilled from a robust teacher using Segmented ARD (Seg-ARD) technique.
]

The evaluation results from this comparison will be able to show the effectiveness of Seg-ARD approach.

#v(2 * line_spacing)

#heading(level: 2)[
  #text(size: 12pt)[Dataset]
]
#v(2 * line_spacing)

The dataset utilized in this study is BDD100K dataset. BDD100K is a large-scale, highly diverse driving video dataset, containing 100,000 high-definition dashboard camera videos, each approximately 40 seconds in duration. To ensure that the dataset represents the high variability of the real world, this dataset contains wide spectrum of environmental conditions, including diverse weather phenomena (rain, snow, and clear skies), varying times of day (night, dawn, and daylight), and distinct geographic locations across urban, highway, and rural settings. In addition, the dataset includes annotations tailored for computer vision tasks, such as object detection and image segmentation (F. Yu et al., 2018).

The selection of BDD100K is justified by the fact that this dataset encompasses real world complexity and scale, compared to Cityscapes dataset. According to F. Yu et al. (2018), due to the difference in the road infrastructure, the model trained using Cityscapes dataset is confused by simple road signs. By capturing various settings and huge brightness diversity, BDD100K bridges the gap between simulated testing and real world deployment.

BDD100K dataset is trusted by many other studies so much so that it has been used in other studies, including object detection (Y. Xie et al., 2025), as well as other studies related to semantic segmentation (Chen et al., 2026; Lee et al., 2026).

For this study, the 10K subset, which consists of 10,000 images will be used. The reason is that the semantic segmentation annotations are only provided for the 10K subset. Although reduced, this subset is twice the size of conventional segmentation benchmarks like Cityscapes (5,000 images) while retaining BDD100K's signature environmental diversity across weather, lighting, and road conditions.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Data Cleaning and Preprocessing]
]
#v(2 * line_spacing)

The first step of data cleaning and preprocessing step focuses on aligning the dimension of the images and the annotated masks. For BDD100K dataset, the original dimension of each image and each mask is expected to be 1280 x 720 pixels, at least according to F. Yu et al. (2018). Nevertheless, during the examination of the dataset, there are 4 images out of 7000 images in the train set that has non-standard dimension. It is best to discard these images, especially when the number of images with non-standard dimension is negligible.

After the alignment of the dimension of the images and masks, a downscaling will be applied. Downscaling the input images and masks to 640 x 360 pixels is necessary for performance reasons, allowing models to train faster and able to segment images without inducing latency more than tolerated.

The next step of preprocessing is to discard the alpha layer found in the mask images. To remove tensor size mismatch between image and mask, which can cause problems during training process, only RGB channels of all image and masks will be preserved, while other channels, such as alpha, will be discarded.

An exploratory data analysis on the mask can also be implemented. In order to understand the composition of each class on all masks, the classification for each pixel also be recorded. From this process, the balance (or rather, imbalance) on each mask and on the overall dataset can be detected. The balance of each class determines the loss function that will be used.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Feature Engineering]
]
#v(2 * line_spacing)

In order for the models to properly use the input images for training and inference, the input tensors are normalized using ImageNet mean and standard deviation, which are $[0.485, 0.456, 0.406]$ and $[0.229, 0.224, 0.225]$ respectively (Gross et al., 2019).

The feature engineering is already handled by the layers on each machine learning model, which are ResNet-18 as the teacher model and MobileNetV2 as the student model. The rationale for the model choices will be explained in the next section. The low-level semantic context, such as edges, textures, and lane markers, are extracted by the early part of the layers, while deeper layers extract high-level semantic context, such as vehicle, sky and road boundaries under diverse lightning conditions.

To improve the feature extraction, data augmentation such as light noise, photometric jittering, and horizontal flip will be used.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[[WIP] Models Used and Knowledge Distillation Procedure]
]
#v(2 * line_spacing)

// TODO: state:
// - loss functions.
// - the random weight sampling.
// - hyperparameters used during training.

Since this study puts an emphasis on the distilled model, therefore both teacher and student models need to be defined as part of the study.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Teacher]
]
#v(2 * line_spacing)

The teacher model will be an encoder-decoder model, with ResNet-18 as the backbone. The reason is because ResNet-18 is a competent model with 11.7 million parameters, enabling the model to contain vast knowledge of the dataset (He et al., 2015). The difference in the parameter count between the teacher model and the student model also enables a careful observation of the difference of the performance between the teacher and the student, specifically the pixel accuracy, intersection over union, and Dice score. These evaluation metrics will be discussed further.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Student]
]
#v(2 * line_spacing)

The student model will be an encoder-decoder model, with MobileNetV2 being the backbone. With only 3.6 million parameters, the latency time when conducting inference in MobileNetV2 is suitable for real-time use cases such as cruise control, whilst being able to store the knowledge distilled from the teacher model.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Knowledge Distillation]
]
#v(2 * line_spacing)

For knowledge distillation process, response-based distillation is chosen. The reason is the difference of the architecture makes it hard to use feature-based distillation. In addition to this, the complexity of the dataset renders the relation-based distillation time consuming and impractical.

For the loss function, in addition to the segmentation loss, this study employes Kullback-Leibler (KL) divergence. As specified in the previous chapter, KL divergence allows the student model to learn the softmax targets of the teacher model.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Adversarial Training]
]
#v(2 * line_spacing)

Since one of the key points of this study is to develop and observe adversarial robustness on distilled segmentation model, therefore a careful choice of attack and defense methods should be discussed.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Attack]
]
#v(2 * line_spacing)

The attack technique to test the robustness is PGD, which is a multi-step iterative process over FGSM (Ayas et al., 2022). PGD will be applied to the input image, with the intention that the attack can yield obfuscation in the segmentation result. If obfuscation is induced, that means the attack method is valid and the experiment on the defense mechanism can be conducted.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Defense]
]
#v(2 * line_spacing)

In order to ensure that the attack against distilled segmentation model can be repelled, or at least minimized, this study proposes Seg-ARD, which is a novel technique to develop defense mechanism on segmentation model. Seg-ARD is based on top of ARD, with modification in the loss function due to the nature of segmentation task. In the original study of ARD, the loss function is a cross-entropy function, because ARD is tested on classification task. However, semantic segmentation task is unable to utilize that loss function.

There are many options when choosing the loss function for semantic segmentation task. However, based on the recommendation from Ma et al. (2021), using Dice-based loss functions generally can yield better results, especially when (but not limited to) dealing with imbalance classes, compared to other loss functions.

In addition, Dice-based loss function for segmentation can be combined with a modified cross-entropy loss. The modified cross-entropy loss is intended to improve the model's accuracy on a pixel basis, since the classification is done per pixel. Various weights will also be tested as part of the comparative study.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Evaluation Metrics]
]
#v(2 * line_spacing)

The evaluation metrics in this study will be comprised of several metrics: pixel accuracy, intersection over union, mean intersection over union, Dice coefficient, attack success rate, and latency.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Pixel Accuracy]
]
#v(2 * line_spacing)

Pixel accuracy is one of the basic methods of evaluating image segmentation models. It basically is the ratio of properly classified pixels, divided by the total number of pixels (Minaee et al., 2021). For $K+1$ classes (where $K$ is the foreground classes and $1$ background class), the pixel accuracy is defined by:
$
  text("PA") = (sum_(i=0)^(K) p_(\i\i)) / (sum_(i=0)^(K) sum_(j=0)^(K) p_(\i\j))
$

#set par(first-line-indent: (amount: 0cm))
where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $p_(\i\i)$ is the number of pixels of class $i$ predicted as belonging to class $i$. In other words, number of pixels in class $i$ that are correctly classified.
  - $p_(\i\j)$ is the number of pixels of class $i$ predicted as belonging to class $j$.
]

#set par(first-line-indent: (amount: tab_length))

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Class-wise Pixel Accuracy]
]
#v(2 * line_spacing)

Class-wise Pixel Accuracy (also equivalent to per-class recall) is an extension of pixel accuracy, in which the ratio of correct pixels are calculated per class. For each class, the pixel accuracy is calculated as follows:
$
  text("CPA")_(i) = (p_(\i\i)) / (sum_(j = 0)^(K) p_(\i\j))
$
#set par(first-line-indent: (amount: 0cm))

where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $p_(\i\i)$ is the number of pixels in class $i$ predicted as class $i$. In other words, this refers to the number of pixels truly belonging in class $i$.
  - $p_(\i\j)$ is the number of pixels in class $i$ predicted as class $j$.
]

#set par(first-line-indent: (amount: tab_length))
Extending this formula yields the formula for mean Pixel Accuracy (Minaee et al., 2021):
$
  text("mPA") = 1 / (K + 1) sum_(i = 0)^(K) text("CPA")_(i)
$

However, for the purpose of the analysis, in this study the class-wise pixel accuracy formula will be used.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Intersection over Union (IoU)]
]
#v(2 * line_spacing)

_Intersection over Union (IoU)_ is defined as the ratio between the area of intersection between the predicted segmentation map $A$ and the ground truth map $B$, divided by the area of union between $A$ and $B$ (Minaee et al., 2021). In other words, IoU detects how closely the area and region of the predicted segmentation match the ground truth.
$
  text("IoU") = (abs(A inter B)) / (abs(A union B))
$

The average of IoU for all classes is called _mean Intersection over Union (mIoU)_. However, for the purpose of the analysis, class-wise IoU will be evaluated, with mIoU as an addition for comparison.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Dice Coefficient]
]
#v(2 * line_spacing)

Dice coefficient is defined as twice the overlap intersection area, divided by total number of pixels.
$
  text("Dice") = (2 abs(A inter B)) / (abs(A) + abs(B))
$

Dice coefficient is widely used in use cases where there's a heavy imbalance in the segmentation. For example, in a brain tumor segmentation, the size of the tumor is tiny compared to the size of the image. However, Dice coefficient also suffers from many limitations, such as segmentation biases and uncertainty ignorance. This is why it is of the utmost importance to pair Dice coefficient with other evaluation metrics, for example IoU (Seghier, 2024).

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Attack Success Rate (ASR)]
]
#v(2 * line_spacing)

Attack success rate is defined as the percentage of adversarial inputs that cause the model to output an incorrect class (Qiao et al., 2026). There are two ways to calculate ASR, either by using targeted ASR, in which the attack measures the model against a specific target output, or untargeted ASR, in which the attack only measures the incorrect rate of the prediction. For untargeted ASR, the original formula for image classification task is defined as:
$
  text("ASR")_(text("untargeted")) = 1 - (text("Acc")_(text("adversarial"))) / (text("Acc")_(text("clean")))
$
#set par(first-line-indent: (amount: 0cm))
where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $text("Acc")_(text("adversarial"))$ is defined as the accuracy of the perturbed inputs.
  - $text("Acc")_(text("clean"))$ is defined as the accuracy of the clean inputs. The clean inputs are inputs from the dataset, without any attack applied.
]

#set par(first-line-indent: (amount: tab_length))

Because this study focuses on semantic segmentation, there should be an adjustment to the formula. Instead of one formula for tracking the degradation of accuracy, now there are 3 formulas focused on each metric. The first formula is to calculate the degradation of mean IoU:
$
  text("Degradation")_(text("mIoU")) = 1 - (text("mIoU")_(text("adversarial"))) / (text("mIoU")_(text("clean")))
$

where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $text("mIoU")_(text("adversarial"))$ is defined as the mean intersection of union (mIoU) for the perturbed inputs. The inputs will be purturbed using multi-step PGD attacks.
  - $text("mIoU")_(text("clean"))$ is defined as the mean intersection of union (mIoU) of the clean inputs. The clean inputs are inputs from the dataset, without any PGD attack applied.
]

The second formula is to calculate the degradation of pixel accuracy:
$
  text("Degradation")_(text("PA")) = 1 - (text("PA")_(text("adversarial"))) / (text("PA")_(text("clean")))
$

where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $text("PA")_(text("adversarial"))$ is defined as the pixel accuracy for the perturbed inputs.
  - $text("PA")_(text("clean"))$ is defined as the pixel accuracy of the clean inputs.
]

The third formula focuses on Dice.
$
  text("Degradation")_(text("Dice")) = 1 - (text("Dice")_(text("adversarial"))) / (text("Dice")_(text("clean")))
$

where:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - $text("Dice")_(text("adversarial"))$ is defined as the Dice score for the perturbed inputs.
  - $text("Dice")_(text("clean"))$ is defined as the Dice score of the clean inputs.
]

Lower ASR is preferrable, signaling high robustnesss that the model possesses.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Latency]
]
#v(2 * line_spacing)

Latency is defined as the total time taken between submitting the input image and receiving the output. In real-time applications, especially in cruise control and self-driving, latency is important because the segmented output is used to assist the driving capability. Higher latency means longer duration needed for the vehicle to take decision, which increases the risk of accident. Therefore, this study aims to lower the latency.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Hardware Utilization and Memory Consumption]
]
#v(2 * line_spacing)

Hardware utilization refers to the workload during inference, while memory consumption refers to the amount of memory that the program used. In the context of real-time autonomous driving systems, medium hardware utilization is preferred, since high utilization may block other systems to run in parallel and higher power consumption, while low utilization indicates that the system is unable to properly benefit from the hardware, which can result in longer latency time.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Prototype Development]
]
#v(2 * line_spacing)

In order to show the robustness capability of the model, a prototype should be developed. The prototype development involves developing a web application that allows users to upload dashboard camera images, and then the machine learning model will develop the segmentation. This allows the users to review the segmentation result.

Excluding the research and development of Seg-ARD, the first iteration of the prototype will be the user interface (UI) and user experience (UX) design. The design of UI and UX should follow.

The second iteration of the prototype will be focused on desgining the frontend of the application. The frontend will be developed using Gradio, allowing seamless integration between the frontend and the backend (Abid et al., 2019), which will host the machine learning model.

The next iteration is to develop the backend. The backend will be also developed using Gradio. The backend will host the distilled segmentation model, and the model will yield the segmentation output that will be displayed in the frontend.

The deployment of such prototype will be done in HuggingFace Space, allowing users from various regions to upload images and test the segmentation capability.

#pagebreak()
#align(center)[
  #heading(level: 1)[
    #text(size: 12pt)[RESULTS]
  ]
]
#v(2 * line_spacing)

In this chapter, the results that correspond with each research question will be presented. The full results can be seen in Google Sheet#footnote[https://docs.google.com/spreadsheets/d/1NVqQEmkcvf-QyhmNNaQxTpI42V0nx__ODbnWGZBqSAY/edit?usp=sharing].

As previously mentioned in Section 3.1 Research Design, the results presented in this chapter belong to four types of students:
\
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  - A non-robust student distilled from a non-robust teacher. Moving forward, this student will be named as "S-NR-V5."
  - A non-robust student distilled from a robust teacher. Moving forward, this student will be named as "S-NR-V6."
  - A robust student distilled from a non-robust teacher, with Segmented ARD (Seg-ARD) used for knowledge distillation. Moving forward, this student will be named as "S-R-V1."
  - A robust student distilled from a robust teacher, with Segmented ARD (Seg-ARD) used for knowledge distillation. Moving forward, this will be named "S-R-V2."
]

In addition, the results of two teachers, a non-robust teacher which will be called "T-NR-V5", and a robust teacher which will be called "T-R-V5", will be presented as well. The results of these two teachers are important to compare the performance, robustness, and latency between the teachers and the students, drawing the conclusions of whether or not the distillation is effective and required.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Training]
]
#v(2 * line_spacing)

This section contains the training graph for all models here. The training graphs function as an indicator of overfitting. If the evaluation loss starts increasing in the middle of the training process, that is the indication of overfitting during the training process.

#figure(
  image("teacher_v5_training_graph.png"),
  caption: [Training graph for non-robust teacher (T-NR-V5).]
)
#v(2 * line_spacing)

#figure(
  image("student_v5_training_graph.png"),
  caption: [Training graph for non-robust student distilled from non-robust teacher (S-NR-V5).]
)
#v(2 * line_spacing)

#figure(
  image("teacher_robust_v5_training_graph.png"),
  caption: [Training graph for robust teacher (T-R-V5).]
)
#v(2 * line_spacing)

#figure(
  image("student_robust_v1_training_graph.png"),
  caption: [Training graph for robust student distilled from non-robust teacher (S-R-V1).]
)
#v(2 * line_spacing)

#figure(
  image("student_robust_v2_training_graph.png"),
  caption: [Training graph for robust student distilled from robust teacher (S-R-V2).]
)
#v(2 * line_spacing)

#figure(
  image("student_v6_training_graph.png"),
  caption: [Training graph for non-robust student distilled from robust teacher (S-NR-V6).]
)
#v(2 * line_spacing)

For all training graphs above, there are no clear indications of overfitting. Some evaluation losses are not able to decrease as much as the training loss, but it is expected, given the unseen data from the evaluation set.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Class Proportion]
]
#v(2 * line_spacing)

In this section, the proportion of each class on all images is calculated. The calculation method is to count the number of pixels associated on the class, then divide by total number of pixels.

#figure(
  table(
    columns: 3,
    table.header(
      [*Class*], [*Proportion*], [*Oversampled?*]
    ),
    [road], [21.45725%], [No],
    [sidewalk], [2.04426%], [Yes],
    [building], [13.32433%], [No],
    [wall], [0.47225%], [Yes],
    [fence], [1.03318%], [Yes],
    [pole], [0.93634%], [Yes],
    [traffic light], [0.18544%], [Yes],
    [traffic sign], [0.34084%], [Yes],
    [vegetation], [13.20344%], [No],
    [terrain], [0.99906%], [Yes],
    [sky], [17.39244%], [No],
    [person], [0.26724%], [Yes],
    [rider], [0.01817%], [Yes],
    [car], [8.06737%], [No],
    [truck], [0.99028%], [Yes],
    [bus], [0.54955%], [Yes],
    [train], [0.01332%], [Yes],
    [motorcycle], [0.02321%], [Yes],
    [bicycle], [0.04879%], [Yes],
    [unknown], [18.63326%], [No],
  ),
  caption: [Class proportion and oversampling decision.]
)
#v(2 * line_spacing)

The class proportion will be used for oversampling process during training process. The training process is conducted by randomly sampling the images based on the given class portions, so not all images were used in one epoch. Images that contain many underrepresented class will occur more often in the training process. This mechanism is provided by PyTorch.

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Segmentation Performance]
]
#v(2 * line_spacing)

The results of the semantic segmentation will be divided into two sub-chapters: accumulated results, and per-class results. In addition, there will be a third sub-chapter for the results of the performance for images with weather and time of day metadata.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Accumulated Results]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*Pixel Accuracy*], [0.9266], [0.9202], [0.8190], [0.8373], [0.7612], [0.9169],
    [*IoU (Micro)*], [0.8633], [0.8521], [0.6934], [0.7201], [0.6145], [0.8465],
    [*Mean IoU*], [0.5291], [0.5166], [0.2329], [0.2988], [0.1704], [0.4709],
    [*Dice*], [0.6205], [0.6096], [0.2874], [0.3854], [0.2064], [0.5597]
  ),
  caption: [Segmentation performance results, accumulated over all classes.]
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Per-Class Results]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9757], [0.9363], [0.9671],
    [sidewalk], [0.7662], [0.6579], [0.7936],
    [building], [0.9223], [0.8365], [0.9109],
    [wall], [0.7364], [0.6666], [0.7999],
    [fence], [0.7098], [0.6242], [0.7687],
    [pole], [0.5037], [0.3614], [0.5309],
    [traffic light], [0.4543], [0.3367], [0.5038],
    [traffic sign], [0.5543], [0.4556], [0.6260],
    [vegetation], [0.9162], [0.8464], [0.9168],
    [terrain], [0.8073], [0.6725], [0.8042],
    [sky], [0.9679], [0.9441], [0.9713],
    [person], [0.7272], [0.4588], [0.6290],
    [rider], [0], [0], [0],
    [car], [0.8763], [0.9354], [0.9340],
    [truck], [0.6955], [0.7969], [0.8204],
    [bus], [0.6835], [0.8515], [0.8120],
    [train], [0], [0], [0],
    [motorcycle], [0.002], [0.002], [0.0003],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of T-NR-V5, grouped by each class.]
)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9741], [0.9305], [0.9640],
    [sidewalk], [0.7553], [0.6359], [0.7774],
    [building], [0.9148], [0.8155], [0.8984],
    [wall], [0.6749], [0.5637], [0.7210],
    [fence], [0.6811], [0.5774], [0.7321],
    [pole], [0.4066], [0.3144], [0.4784],
    [traffic light], [0.3823], [0.3085], [0.4715],
    [traffic sign], [0.5004], [0.4304], [0.6018],
    [vegetation], [0.9176], [0.8328], [0.9088],
    [terrain], [0.7511], [0.6356], [0.7772],
    [sky], [0.9611], [0.9384], [0.9682],
    [person], [0.6637], [0.4965], [0.6635],
    [rider], [0], [0], [0],
    [car], [0.9354], [0.8678], [0.9292],
    [truck], [0.7969], [0.6997], [0.8233],
    [bus], [0.8515], [0.7675], [0.8685],
    [train], [0], [0], [0],
    [motorcycle], [0], [0], [0],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of S-NR-V5, grouped by each class.]
)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9554], [0.8360], [0.9107],
    [sidewalk], [0.3881], [0.2781], [0.4351],
    [building], [0.8166], [0.6202], [0.7656],
    [wall], [0], [0], [0],
    [fence], [0.0644], [0.0598], [0.1128],
    [pole], [0.0789], [0.0722], [0.1346],
    [traffic light], [0], [0], [0],
    [traffic sign], [0], [0], [0],
    [vegetation], [0.8164], [0.6457], [0.7847],
    [terrain], [0.1711], [0.1466], [0.2557],
    [sky], [0.9428], [0.9031], [0.9491],
    [person], [0], [0], [0],
    [rider], [0], [0], [0],
    [car], [0.8299], [0.6906], [0.8170],
    [truck], [0.2496], [0.1732], [0.2953],
    [bus], [0], [0], [0],
    [train], [0], [0], [0],
    [motorcycle], [0], [0], [0],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of T-R-V5, grouped by each class.]
)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9666], [0.8608], [0.9252],
    [sidewalk], [0.3402], [0.2904], [0.4501],
    [building], [0.9159], [0.6362], [0.7776],
    [wall], [0.0210], [0.0209], [0.0408],
    [fence], [0.2616], [0.2118], [0.3496],
    [pole], [0.1926], [0.1594], [0.2750],
    [traffic light], [0.0120], [0.0119], [0.0236],
    [traffic sign], [0.2587], [0.2003], [0.3338],
    [vegetation], [0.7113], [0.6556], [0.7920],
    [terrain], [0.2132], [0.1956], [0.3271],
    [sky], [0.9585], [0.9081], [0.9518],
    [person], [0.2482], [0.1879], [0.3163],
    [rider], [0], [0], [0],
    [car], [0.8757], [0.7534], [0.8594],
    [truck], [0.3035], [0.2415], [0.3891],
    [bus], [0.4200], [0.3436], [0.5115],
    [train], [0], [0], [0],
    [motorcycle], [0], [0], [0],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of S-R-V1, grouped by each class.]
)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9716], [0.7550], [0.8604],
    [sidewalk], [0.0075], [0.0075], [0.0148],
    [building], [0.6109], [0.4808], [0.6494],
    [wall], [0], [0], [0],
    [fence], [0.0004], [0.0004], [0.0009],
    [pole], [0.0095], [0.0095], [0.0187],
    [traffic light], [0], [0], [0],
    [traffic sign], [0.0018], [0.0018], [0.0036],
    [vegetation], [0.8383], [0.5743], [0.7296],
    [terrain], [0.0127], [0.0126], [0.0249],
    [sky], [0.9538], [0.8730], [0.9322],
    [person], [0], [0], [0],
    [rider], [0], [0], [0],
    [car], [0.6447], [0.5227], [0.6865],
    [truck], [0], [0], [0.0001],
    [bus], [0], [0], [0],
    [train], [0], [0], [0],
    [motorcycle], [0], [0], [0],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of S-R-V2, grouped by each class.]
)

#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Pixel Accuracy*], [*IoU*], [*Dice*]
    ),
    [road], [0.9757], [0.9287], [0.9287],
    [sidewalk], [0.7149], [0.6097], [0.7575],
    [building], [0.9105], [0.8205], [0.9014],
    [wall], [0.6706], [0.5651], [0.7221],
    [fence], [0.6409], [0.5248], [0.6883],
    [pole], [0.3456], [0.2792], [0.4365],
    [traffic light], [0], [0], [0],
    [traffic sign], [0.4054], [0.3412], [0.5087],
    [vegetation], [0.9237], [0.8275], [0.9056],
    [terrain], [0.7354], [0.5980], [0.7485],
    [sky], [0.9679], [0.9389], [0.9685],
    [person], [0.5058], [0.4002], [0.5716],
    [rider], [0], [0], [0],
    [car], [0.9345], [0.8704], [0.9307],
    [truck], [0.7596], [0.6103], [0.7580],
    [bus], [0.7081], [0.6321], [0.7746],
    [train], [0], [0], [0],
    [motorcycle], [0], [0], [0],
    [bicycle], [0], [0], [0],
  ),
  caption: [Segmentation performance results of S-NR-V6, grouped by each class.]
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Results Based on Weather and Time of Day]
] <segmentation-performance-weather-timeofday>
#v(2 * line_spacing)

Due to the limitation of the dataset, not all images have weather and time of day metadata. The dataset used for this study belongs to the 10K subset, as previously stated. However, some images and masks in the 10K subset are not present in the 100K dataset, so it is not completely a subset, rather it forms an intersection. Moreover, some images have weather and/or time of day metadata, but the value is "unknown", and hence these images need to be filtered out. Due to this reason, only 545 images have valid weather and time of day metadata, all of which are used for evaluation in this section.

#figure(
  table(
    columns: 3,
    table.header(
      [*Weather*], [*Time of day*], [*Count*]
    ),
    [clear], [daytime], [176],
    [overcast], [daytime], [112],
    [partly cloudy], [daytime], [68],
    [rainy], [daytime], [47],
    [snowy], [daytime], [46],
    [clear], [night], [29],
    [clear], [dawn/dusk], [26],
    [overcast], [dawn/dusk], [16],
    [partly cloudy], [dawn/dusk], [8],
    [snowy], [dawn/dusk], [8],
    [rainy], [dawn/dusk], [6],
    [snowy], [night], [2],
    [rainy], [night], [1],
  ),
  caption: [Distribution of weather and time of day group.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Teacher - Non-Robust - V5 (T-NR-V5)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.9132], [0.8403], [0.5167], [0.6102],
    [overcast], [daytime], [0.9152], [0.8437], [0.4850], [0.5782],
    [partly cloudy], [daytime], [0.9064], [0.8288], [0.5257], [0.6101],
    [rainy], [daytime], [0.9080], [0.8315], [0.6172], [0.7109],
    [snowy], [daytime], [0.9063], [0.8287], [0.5576], [0.6529],
    [clear], [night], [0.8705], [0.7707], [0.4682], [0.5617],
    [clear], [dawn/dusk], [0.9295], [0.8682], [0.4954], [0.5870],
    [overcast], [dawn/dusk], [0.9482], [0.9015], [0.6400], [0.7237],
    [partly cloudy], [dawn/dusk], [0.8445], [0.7308], [0.5285], [0.6208],
    [snowy], [dawn/dusk], [0.9363], [0.8803], [0.5401], [0.6020],
    [rainy], [dawn/dusk], [0.8311], [0.7111], [0.4913], [0.5615],
    [snowy], [night], [0.8293], [0.7083], [0.5341], [0.5837],
    [rainy], [night], [0.9547], [0.9133], [0.6832], [0.7006],
  ),
  caption: [Performance results of T-NR-V5, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Non-Robust - V5 (S-NR-V5)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.9077], [0.8310], [0.5078], [0.6027],
    [overcast], [daytime], [0.9113], [0.8370], [0.4895], [0.5821],
    [partly cloudy], [daytime], [0.8977], [0.8144], [0.5129], [0.5954],
    [rainy], [daytime], [0.9102], [0.8353], [0.6270], [0.7189],
    [snowy], [daytime], [0.8840], [0.7922], [0.5132], [0.6123],
    [clear], [night], [0.8116], [0.6830], [0.4144], [0.5087],
    [clear], [dawn/dusk], [0.9249], [0.8604], [0.5029], [0.5958],
    [overcast], [dawn/dusk], [0.9408], [0.8882], [0.6541], [0.7379],
    [partly cloudy], [dawn/dusk], [0.8495], [0.7384], [0.5586], [0.6493],
    [snowy], [dawn/dusk], [0.9330], [0.8743], [0.5083], [0.5698],
    [rainy], [dawn/dusk], [0.8400], [0.7242], [0.4401], [0.5142],
    [snowy], [night], [0.8376], [0.7206], [0.5404], [0.5806],
    [rainy], [night], [0.9716], [0.9447], [0.6895], [0.7060],
  ),
  caption: [Performance results of S-NR-V5, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Teacher - Robust - V5 (T-R-V5)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.8051], [0.6738], [0.2355], [0.2937],
    [overcast], [daytime], [0.8338], [0.7149], [0.2476], [0.3008],
    [partly cloudy], [daytime], [0.7889], [0.6513], [0.2802], [0.3349],
    [rainy], [daytime], [0.8042], [0.6725], [0.3389], [0.3948],
    [snowy], [daytime], [0.7607], [0.6138], [0.2490], [0.2966],
    [clear], [night], [0.6646], [0.4977], [0.2231], [0.2658],
    [clear], [dawn/dusk], [0.8262], [0.7038], [0.2634], [0.3091],
    [overcast], [dawn/dusk], [0.8415], [0.7264], [0.4038], [0.4585],
    [partly cloudy], [dawn/dusk], [0.7199], [0.5624], [0.2761], [0.3250],
    [snowy], [dawn/dusk], [0.8365], [0.7190], [0.4178], [0.4565],
    [rainy], [dawn/dusk], [0.5891], [0.4176], [0.3383], [0.3852],
    [snowy], [night], [0.7501], [0.6001], [0.5264], [0.5473],
    [rainy], [night], [0.5688], [0.3974], [0.6208], [0.6393],
  ),
  caption: [Performance results of T-R-V5, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Robust - V1 (S-R-V1)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.8238], [0.7003], [0.2875], [0.3713],
    [overcast], [daytime], [0.8400], [0.7241], [0.2941], [0.3752],
    [partly cloudy], [daytime], [0.8124], [0.6841], [0.3282], [0.4033],
    [rainy], [daytime], [0.8284], [0.7071], [0.4087], [0.4887],
    [snowy], [daytime], [0.7824], [0.6426], [0.2985], [0.3754],
    [clear], [night], [0.6825], [0.5180], [0.2475], [0.3020],
    [clear], [dawn/dusk], [0.8509], [0.7405], [0.3004], [0.3579],
    [overcast], [dawn/dusk], [0.8769], [0.7808], [0.4547], [0.5289],
    [partly cloudy], [dawn/dusk], [0.7605], [0.6135], [0.3321], [0.4020],
    [snowy], [dawn/dusk], [0.8456], [0.7324], [0.3698], [0.4107],
    [rainy], [dawn/dusk], [0.7122], [0.5530], [0.3894], [0.4360],
    [snowy], [night], [0.7506], [0.6008], [0.5260], [0.5493],
    [rainy], [night], [0.6996], [0.5380], [0.6565], [0.6816],
  ),
  caption: [Performance results of S-R-V1, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Robust - V2 (S-R-V1)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.7477], [0.5971], [0.1670], [0.2040],
    [overcast], [daytime], [0.7732], [0.6302], [0.1790], [0.2154],
    [partly cloudy], [daytime], [0.7213], [0.5641], [0.2153], [0.2526],
    [rainy], [daytime], [0.7496], [0.5995], [0.2778], [0.3128],
    [snowy], [daytime], [0.7066], [0.5463], [0.2072], [0.2456],
    [clear], [night], [0.5578], [0.3868], [0.1983], [0.2356],
    [clear], [dawn/dusk], [0.8003], [0.6671], [0.2301], [0.2633],
    [overcast], [dawn/dusk], [0.8223], [0.6983], [0.3376], [0.3701],
    [partly cloudy], [dawn/dusk], [0.7246], [0.5681], [0.2594], [0.2988],
    [snowy], [dawn/dusk], [0.7401], [0.5875], [0.4720], [0.5058],
    [rainy], [dawn/dusk], [0.5339], [0.3641], [0.3735], [0.4113],
    [snowy], [night], [0.6161], [0.4452], [0.4006], [0.4255],
    [rainy], [night], [0.4315], [0.2751], [0.6122], [0.6235],
  ),
  caption: [Performance results of S-R-V2, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Non-Robust - V6 (S-NR-V6)]
]
#v(2 * line_spacing)

#figure(
  table(
    columns: 6,
    table.header(
      [*Weather*], [*Time of day*], [*Pixel Accuracy*], [*IoU (Micro)*], [*IoU (Macro)*], [*Dice*]
    ),
    [clear], [daytime], [0.9038], [0.8245], [0.4530], [0.5436],
    [overcast], [daytime], [0.9052], [0.8267], [0.4497], [0.5414],
    [partly cloudy], [daytime], [0.8900], [0.8018], [0.4819], [0.5631],
    [rainy], [daytime], [0.9060], [0.8282], [0.5693], [0.6532],
    [snowy], [daytime], [0.8751], [0.7780], [0.4557], [0.5434],
    [clear], [night], [0.8413], [0.7260], [0.3891], [0.4687],
    [clear], [dawn/dusk], [0.9265], [0.8630], [0.4684], [0.5397],
    [overcast], [dawn/dusk], [0.9463], [0.8980], [0.6359], [0.7084],
    [partly cloudy], [dawn/dusk], [0.8568], [0.7495], [0.4573], [0.5299],
    [snowy], [dawn/dusk], [0.9467], [0.8987], [0.6297], [0.6824],
    [rainy], [dawn/dusk], [0.7993], [0.6657], [0.4513], [0.5067],
    [snowy], [night], [0.7947], [0.6594], [0.4981], [0.5216],
    [rainy], [night], [0.9646], [0.9317], [0.7951], [0.8100],
  ),
  caption: [Performance results of S-NR-V6, grouped by weather and time of day.]
)

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Adversarial Robustness]
]
#v(2 * line_spacing)

The results of the adversarial robustness will be divided into two sub-chapters: accumulated results, and per-class results. In addition, there will be a third sub-chapter for the results of the robustness for images with weather and time of day metadata.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Accumulated Robustness Results]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*Clean Pixel Accuracy*], [0.9266], [0.9202], [0.8190], [0.8373], [0.7612], [0.9169],
    [*Adversarial Pixel Accuracy*], [0.0359], [0.0152], [0.7220], [0.6994], [0.5909], [0.0086],
    [*Degradation of Pixel Accuracy*], [96.12%], [98.35%], [11.84%], [16.47%], [22.38%], [99.06%],
  ),
  caption: [Accumulated robustness results, measured by pixel accuracy.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*Clean Mean IoU*], [0.5291], [0.5166], [0.2329], [0.2988], [0.1704], [0.4709],
    [*Adversarial Mean IoU*], [0.0101], [0.0058], [0.1774], [0.1842], [0.1117], [0.0026],
    [*Degradation of Mean IoU*], [98.10%], [98.88%], [23.82%], [38.37%], [34.48%], [99.46%],
  ),
  caption: [Accumulated robustness results, measured by mean IoU.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*Clean Dice*], [0.6205], [0.6097], [0.2874], [0.3854], [0.2064], [0.5597],
    [*Adversarial Dice*], [0.0197], [0.0114], [0.2284], [0.2478], [0.1465], [0.0051],
    [*Degradation of Dice*], [96.82%], [98.14%], [20.52%], [35.69%], [29.01%], [99.09%],
  ),
  caption: [Accumulated robustness results, measured by Dice score.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Per-Class Results]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.9363], [0.0100], [98.93%],
    [sidewalk], [0.6579], [0.0078], [98.82%],
    [building], [0.8365], [0.0297], [96.45%],
    [wall], [0.6666], [0.0002], [99.96%],
    [fence], [0.6242], [0.0052], [99.17%],
    [pole], [0.3614], [0.0117], [96.77%],
    [traffic light], [0.3367], [0.0135], [96.00%],
    [traffic sign], [0.4556], [0.0062], [98.64%],
    [vegetation], [0.8464], [0.0287], [96.61%],
    [terrain], [0.6725], [0.0077], [98.85%],
    [sky], [0.9441], [0.0108], [98.85%],
    [person], [0.4588], [0.0110], [97.60%],
    [rider], [0], [0], [0.00%],
    [car], [0.8763], [0.0272], [96.90%],
    [truck], [0.6955], [0.0014], [99.80%],
    [bus], [0.6835], [0.0201], [97.07%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0.0002], [0], [100.00%],
    [bicycle], [5.19 #math.times 10^(-6)], [0], [100.00%]
  ),
  caption: [Segmentation robustness results of T-NR-V5, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.9305], [0.0055], [99.41%],
    [sidewalk], [0.6359], [0.0055], [99.13%],
    [building], [0.8155], [0.0245], [96.99%],
    [wall], [0.5637], [0.0004], [99.93%],
    [fence], [0.5773], [0.0047], [99.19%],
    [pole], [0.3144], [0.0015], [99.51%],
    [traffic light], [0.3085], [0.0059], [98.10%],
    [traffic sign], [0.4304], [0.0032], [99.25%],
    [vegetation], [0.8328], [0.0071], [99.15%],
    [terrain], [0.6356], [0.0040], [99.37%],
    [sky], [0.9384], [0.0026], [99.72%],
    [person], [0.4965], [0.0109], [97.80%],
    [rider], [0], [0], [0.00%],
    [car], [0.8678], [0.0074], [99.15%],
    [truck], [0.6998], [0.0005], [99.93%],
    [bus], [0.7676], [0.0258], [96.64%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0], [0], [0.00%],
    [bicycle], [0], [0], [0.00%],
  ),
  caption: [Segmentation robustness results of S-NR-V5, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: true
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.8360], [0.7574], [9.39%],
    [sidewalk], [0.2781], [0.1485], [46.61%],
    [building], [0.6202], [0.4317], [30.40%],
    [wall], [0], [0], [0.00%],
    [fence], [0.0598], [0.0291], [51.34%],
    [pole], [0.0722], [0.0657], [8.97%],
    [traffic light], [0], [0], [0.00%],
    [traffic sign], [0], [0], [0.00%],
    [vegetation], [0.6457], [0.4671], [27.65%],
    [terrain], [0.1466], [0.0418], [71.47%],
    [sky], [0.9031], [0.8596], [4.81%],
    [person], [0], [0], [0.00%],
    [rider], [0], [0], [0.00%],
    [car], [0.6906], [0.5034], [27.10%],
    [truck], [0.1732], [0.0667], [61.48%],
    [bus], [0], [0], [0.00%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0], [0], [0.00%],
    [bicycle], [0], [0], [0.00%],
  ),
  caption: [Segmentation robustness results of T-R-V5, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.8607], [0.7632], [11.33%],
    [sidewalk], [0.2903], [0.0957], [67.04%],
    [building], [0.6362], [0.4170], [34.45%],
    [wall], [0.0208], [0.0027], [87.23%],
    [fence], [0.2118], [0.0423], [80.02%],
    [pole], [0.1594], [0.1170], [26.56%],
    [traffic light], [0.0119], [0.0057], [52.08%],
    [traffic sign], [0.2003], [0.1087], [45.71%],
    [vegetation], [0.6557], [0.3401], [48.14%],
    [terrain], [0.1956], [0.0340], [82.64%],
    [sky], [0.9081], [0.8512], [6.26%],
    [person], [0.1878], [0.0748], [60.18%],
    [rider], [0], [0], [0.00%],
    [car], [0.7534], [0.4897], [35.00%],
    [truck], [0.2415], [0.0435], [81.97%],
    [bus], [0.3436], [0.1134], [67.01%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0], [0], [0.00%],
    [bicycle], [0], [0], [0.00%],
  ),
  caption: [Segmentation robustness results of S-R-V1, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.7551], [0.6304], [16.52%],
    [sidewalk], [0.0075], [0.0037], [50.56%],
    [building], [0.4807], [0.1680], [65.05%],
    [wall], [0], [2.45 #math.times 10^(-6)], [0.00%],
    [fence], [0.0004], [8.00 #math.times 10^(-5)], [81.98%],
    [pole], [0.0094], [0.0023], [75.52%],
    [traffic light], [0], [0], [0.00%],
    [traffic sign], [0.0018], [6.48 #math.times 10^(-5)], [96.39%],
    [vegetation], [0.5743], [0.3156], [45.04%],
    [terrain], [0.0126], [0.0003], [97.38%],
    [sky], [0.8730], [0.7917], [9.31%],
    [person], [0], [0], [0.00%],
    [rider], [0], [0], [0.00%],
    [car], [0.5229], [0.2093], [59.97%],
    [truck], [2.80 #math.times 10^(-5)], [3.45 #math.times 10^(-7)], [98.77%],
    [bus], [0], [0], [0.00%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0], [0], [0.00%],
    [bicycle], [0], [0], [0.00%],
  ),
  caption: [Segmentation robustness results of S-R-V2, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 4,
    table.header(
      [*Class*], [*Clean IoU*], [*Adversarial IoU*], [*Degradation*]
    ),
    [road], [0.9287], [0.0050], [99.46%],
    [sidewalk], [0.6096], [0.0014], [99.77%],
    [building], [0.8205], [0.0034], [99.58%],
    [wall], [0.5651], [0.0002], [99.96%],
    [fence], [0.5247], [0.0018], [99.65%],
    [pole], [0.2792], [0.0043], [98.45%],
    [traffic light], [0], [0], [0.00%],
    [traffic sign], [0.3412], [0.0048], [98.59%],
    [vegetation], [0.8275], [0.0070], [99.16%],
    [terrain], [0.5981], [0.0022], [99.63%],
    [sky], [0.9389], [0.0138], [98.53%],
    [person], [0.4002], [0.0016], [99.60%],
    [rider], [0], [0], [0.00%],
    [car], [0.8704], [0.0012], [99.87%],
    [truck], [0.6103], [0.0010], [99.83%],
    [bus], [0.6321], [0.0008], [99.87%],
    [train], [0], [0], [0.00%],
    [motorcycle], [0], [0], [0.00%],
    [bicycle], [0], [0], [0.00%],
  ),
  caption: [Segmentation robustness results of S-NR-V6, grouped by each class.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Results Based on Weather and Time of Day]
]
#v(2 * line_spacing)

Similar to the limitation mentioned previously (in @segmentation-performance-weather-timeofday), the adversarial robustness results of this part will not be based on all 10K images, but rather a small selected subset of 545 images.

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Teacher - Non-Robust - V5 (T-NR-V5)]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.9132], [0.0392], [95.71%],
    [overcast], [daytime], [0.9152], [0.0355], [96.12%],
    [partly cloudy], [daytime], [0.9064], [0.0278], [96.93%],
    [rainy], [daytime], [0.9080], [0.0263], [97.11%],
    [snowy], [daytime], [0.9063], [0.0488], [94.61%],
    [clear], [night], [0.8705], [0.0209], [97.60%],
    [clear], [dawn/dusk], [0.9295], [0.0446], [95.20%],
    [overcast], [dawn/dusk], [0.9482], [0.0283], [97.01%],
    [partly cloudy], [dawn/dusk], [0.8445], [0.0176], [97.92%],
    [snowy], [dawn/dusk], [0.9364], [0.0369], [96.06%],
    [rainy], [dawn/dusk], [0.8311], [0.0339], [95.92%],
    [snowy], [night], [0.8293], [0.0178], [97.85%],
    [rainy], [night], [0.9547], [0.0574], [93.99%],
  ),
  caption: [Pixel accuracy-based robustness results of T-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.5168], [0.0114], [97.79%],
    [overcast], [daytime], [0.4850], [0.0105], [97.84%],
    [partly cloudy], [daytime], [0.4994], [0.0073], [98.53%],
    [rainy], [daytime], [0.5722], [0.0067], [98.83%],
    [snowy], [daytime], [0.5331], [0.0139], [97.39%],
    [clear], [night], [0.4644], [0.0054], [98.84%],
    [clear], [dawn/dusk], [0.4673], [0.0100], [97.87%],
    [overcast], [dawn/dusk], [0.6348], [0.0082], [98.71%],
    [partly cloudy], [dawn/dusk], [0.4731], [0.0056], [98.82%],
    [snowy], [dawn/dusk], [0.5585], [0.0133], [97.61%],
    [rainy], [dawn/dusk], [0.4525], [0.0131], [97.10%],
    [snowy], [night], [0.4610], [0.0056], [98.79%],
    [rainy], [night], [0.5960], [0.0204], [96.58%],
  ),
  caption: [Mean IoU-based robustness results of T-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.6103], [0.0224], [96.33%],
    [overcast], [daytime], [0.5782], [0.0205], [96.45%],
    [partly cloudy], [daytime], [0.5885], [0.0144], [97.55%],
    [rainy], [daytime], [0.6769], [0.0132], [98.05%],
    [snowy], [daytime], [0.6336], [0.0268], [95.78%],
    [clear], [night], [0.5689], [0.0106], [98.14%],
    [clear], [dawn/dusk], [0.5639], [0.0192], [96.59%],
    [overcast], [dawn/dusk], [0.7341], [0.0159], [97.83%],
    [partly cloudy], [dawn/dusk], [0.5762], [0.0108], [98.12%],
    [snowy], [dawn/dusk], [0.6490], [0.0250], [96.14%],
    [rainy], [dawn/dusk], [0.5477], [0.0249], [95.46%],
    [snowy], [night], [0.5657], [0.0108], [98.09%],
    [rainy], [night], [0.6621], [0.0370], [94.41%],
  ),
  caption: [Dice-based robustness results of T-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Non-Robust - V5 (S-NR-V5)]
]

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.9077], [0.0154], [98.31%],
    [overcast], [daytime], [0.9113], [0.0136], [98.50%],
    [partly cloudy], [daytime], [0.8977], [0.0158], [98.24%],
    [rainy], [daytime], [0.9102], [0.0096], [98.94%],
    [snowy], [daytime], [0.8840], [0.0235], [97.34%],
    [clear], [night], [0.8117], [0.0091], [98.88%],
    [clear], [dawn/dusk], [0.9249], [0.0160], [98.27%],
    [overcast], [dawn/dusk], [0.9408], [0.0126], [98.66%],
    [partly cloudy], [dawn/dusk], [0.8495], [0.0053], [99.38%],
    [snowy], [dawn/dusk], [0.9330], [0.0179], [98.08%],
    [rainy], [dawn/dusk], [0.8402], [0.0066], [99.22%],
    [snowy], [night], [0.8377], [0.0152], [98.18%],
    [rainy], [night], [0.9716], [0.0011], [99.88%],
  ),
  caption: [Pixel accuracy-based robustness results of S-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.5078], [0.0047], [99.08%],
    [overcast], [daytime], [0.4895], [0.0045], [99.09%],
    [partly cloudy], [daytime], [0.4859], [0.0044], [99.10%],
    [rainy], [daytime], [0.5832], [0.0031], [99.47%],
    [snowy], [daytime], [0.4863], [0.0054], [98.90%],
    [clear], [night], [0.4045], [0.0015], [99.63%],
    [clear], [dawn/dusk], [0.4753], [0.0045], [99.05%],
    [overcast], [dawn/dusk], [0.5892], [0.0033], [99.44%],
    [partly cloudy], [dawn/dusk], [0.5067], [0.0021], [99.58%],
    [snowy], [dawn/dusk], [0.5122], [0.0088], [98.28%],
    [rainy], [dawn/dusk], [0.4546], [0.0031], [99.32%],
    [snowy], [night], [0.3632], [0.0053], [98.53%],
    [rainy], [night], [0.6201], [0.0034], [99.45%],
  ),
  caption: [Mean IoU-based robustness results of S-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.6027], [0.0092], [98.47%],
    [overcast], [daytime], [0.5821], [0.0088], [98.49%],
    [partly cloudy], [daytime], [0.5729], [0.0086], [98.49%],
    [rainy], [daytime], [0.6859], [0.0061], [99.11%],
    [snowy], [daytime], [0.5908], [0.0105], [98.22%],
    [clear], [night], [0.5098], [0.0030], [99.41%],
    [clear], [dawn/dusk], [0.5733], [0.0089], [98.45%],
    [overcast], [dawn/dusk], [0.6888], [0.0064], [99.06%],
    [partly cloudy], [dawn/dusk], [0.6081], [0.0042], [99.31%],
    [snowy], [dawn/dusk], [0.6021], [0.0171], [97.16%],
    [rainy], [dawn/dusk], [0.5551], [0.0060], [98.92%],
    [snowy], [night], [0.4480], [0.0103], [97.70%],
    [rainy], [night], [0.6829], [0.0067], [99.02%],
  ),
  caption: [Dice-based robustness results of S-NR-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Teacher - Robust - V5 (T-R-V5)]
]

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.8051], [0.7038], [12.59%],
    [overcast], [daytime], [0.8338], [0.7403], [11.21%],
    [partly cloudy], [daytime], [0.7889], [0.6914], [12.35%],
    [rainy], [daytime], [0.8042], [0.6805], [15.38%],
    [snowy], [daytime], [0.7607], [0.6421], [15.59%],
    [clear], [night], [0.6646], [0.5753], [13.44%],
    [clear], [dawn/dusk], [0.8262], [0.7446], [9.87%],
    [overcast], [dawn/dusk], [0.8415], [0.7359], [12.55%],
    [partly cloudy], [dawn/dusk], [0.7199], [0.6312], [12.32%],
    [snowy], [dawn/dusk], [0.8365], [0.7015], [16.14%],
    [rainy], [dawn/dusk], [0.5891], [0.4211], [28.53%],
    [snowy], [night], [0.7501], [0.6240], [16.81%],
    [rainy], [night], [0.5687], [0.4545], [20.09%],
  ),
  caption: [Pixel accuracy-based robustness results of T-R-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.2355], [0.1759], [25.32%],
    [overcast], [daytime], [0.2476], [0.1890], [23.67%],
    [partly cloudy], [daytime], [0.2402], [0.1781], [25.85%],
    [rainy], [daytime], [0.2612], [0.1845], [29.35%],
    [snowy], [daytime], [0.2072], [0.1593], [23.12%],
    [clear], [night], [0.1317], [0.0961], [27.08%],
    [clear], [dawn/dusk], [0.2224], [0.1772], [20.34%],
    [overcast], [dawn/dusk], [0.2920], [0.2150], [26.38%],
    [partly cloudy], [dawn/dusk], [0.1909], [0.1483], [22.32%],
    [snowy], [dawn/dusk], [0.3030], [0.2208], [27.13%],
    [rainy], [dawn/dusk], [0.1733], [0.1131], [34.78%],
    [snowy], [night], [0.2225], [0.1615], [27.42%],
    [rainy], [night], [0.3592], [0.3164], [11.91%],
  ),
  caption: [Mean IoU-based robustness results of T-R-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.2937], [0.2290], [22.03%],
    [overcast], [daytime], [0.3008], [0.2404], [20.10%],
    [partly cloudy], [daytime], [0.2979], [0.2288], [23.21%],
    [rainy], [daytime], [0.3236], [0.2382], [26.39%],
    [snowy], [daytime], [0.2575], [0.2116], [17.82%],
    [clear], [night], [0.1794], [0.1354], [24.51%],
    [clear], [dawn/dusk], [0.2707], [0.2255], [16.70%],
    [overcast], [dawn/dusk], [0.3570], [0.2837], [20.55%],
    [partly cloudy], [dawn/dusk], [0.2456], [0.1998], [18.64%],
    [snowy], [dawn/dusk], [0.3595], [0.2774], [22.83%],
    [rainy], [dawn/dusk], [0.2370], [0.1699], [28.32%],
    [snowy], [night], [0.2665], [0.2150], [19.34%],
    [rainy], [night], [0.4294], [0.3781], [11.96%],
  ),
  caption: [Dice-based robustness results of T-R-V5, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Robust - V1 (S-R-V1)]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.8238], [0.6838], [17.00%],
    [overcast], [daytime], [0.8400], [0.7152], [14.86%],
    [partly cloudy], [daytime], [0.8124], [0.6863], [15.53%],
    [rainy], [daytime], [0.8284], [0.6593], [20.41%],
    [snowy], [daytime], [0.7824], [0.6430], [17.82%],
    [clear], [night], [0.6825], [0.5498], [19.44%],
    [clear], [dawn/dusk], [0.8509], [0.7035], [17.32%],
    [overcast], [dawn/dusk], [0.8769], [0.7411], [15.49%],
    [partly cloudy], [dawn/dusk], [0.7605], [0.6074], [20.13%],
    [snowy], [dawn/dusk], [0.8456], [0.6859], [18.88%],
    [rainy], [dawn/dusk], [0.7122], [0.4363], [38.74%],
    [snowy], [night], [0.7506], [0.5115], [31.86%],
    [rainy], [night], [0.6995], [0.4457], [36.29%],
  ),
  caption: [Pixel accuracy-based robustness results of S-R-V1, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.2875], [0.1760], [38.78%],
    [overcast], [daytime], [0.2941], [0.1852], [37.04%],
    [partly cloudy], [daytime], [0.2909], [0.1894], [34.90%],
    [rainy], [daytime], [0.3391], [0.1948], [42.54%],
    [snowy], [daytime], [0.2595], [0.1617], [37.69%],
    [clear], [night], [0.1590], [0.0863], [45.73%],
    [clear], [dawn/dusk], [0.2615], [0.1724], [34.07%],
    [overcast], [dawn/dusk], [0.3524], [0.2135], [39.44%],
    [partly cloudy], [dawn/dusk], [0.2536], [0.1686], [33.50%],
    [snowy], [dawn/dusk], [0.3097], [0.1966], [36.52%],
    [rainy], [dawn/dusk], [0.2427], [0.1214], [49.97%],
    [snowy], [night], [0.2218], [0.1217], [45.13%],
    [rainy], [night], [0.4945], [0.3048], [38.35%],
  ),
  caption: [Mean IoU-based robustness results of S-R-V1, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.3712], [0.2364], [36.31%],
    [overcast], [daytime], [0.3752], [0.2424], [35.38%],
    [partly cloudy], [daytime], [0.3702], [0.2513], [32.10%],
    [rainy], [daytime], [0.4285], [0.2626], [38.72%],
    [snowy], [daytime], [0.3407], [0.2168], [36.38%],
    [clear], [night], [0.2200], [0.1223], [44.41%],
    [clear], [dawn/dusk], [0.3222], [0.2273], [29.45%],
    [overcast], [dawn/dusk], [0.4406], [0.2786], [36.76%],
    [partly cloudy], [dawn/dusk], [0.3317], [0.2349], [29.19%],
    [snowy], [dawn/dusk], [0.3694], [0.2520], [31.79%],
    [rainy], [dawn/dusk], [0.3060], [0.1719], [43.83%],
    [snowy], [night], [0.2710], [0.1763], [34.96%],
    [rainy], [night], [0.5901], [0.3964], [32.82%],
  ),
  caption: [Dice-based robustness results of S-R-V1, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Robust - V2 (S-R-V2)]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.7477], [0.5878], [21.39%],
    [overcast], [daytime], [0.7732], [0.5964], [22.86%],
    [partly cloudy], [daytime], [0.7213], [0.5685], [21.18%],
    [rainy], [daytime], [0.7496], [0.5211], [30.48%],
    [snowy], [daytime], [0.7066], [0.5158], [27.00%],
    [clear], [night], [0.5577], [0.3142], [43.66%],
    [clear], [dawn/dusk], [0.8003], [0.6600], [17.53%],
    [overcast], [dawn/dusk], [0.8223], [0.6553], [20.32%],
    [partly cloudy], [dawn/dusk], [0.7246], [0.5468], [24.54%],
    [snowy], [dawn/dusk], [0.7401], [0.5349], [27.72%],
    [rainy], [dawn/dusk], [0.5338], [0.2831], [46.97%],
    [snowy], [night], [0.6158], [0.2140], [65.26%],
    [rainy], [night], [0.4313], [0.2979], [30.93%],
  ),
  caption: [Pixel accuracy-based robustness results of S-R-V2, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.1670], [0.1106], [33.75%],
    [overcast], [daytime], [0.1790], [0.1175], [34.40%],
    [partly cloudy], [daytime], [0.1717], [0.1149], [33.06%],
    [rainy], [daytime], [0.1928], [0.1142], [40.76%],
    [snowy], [daytime], [0.1632], [0.1023], [37.28%],
    [clear], [night], [0.1039], [0.0445], [57.21%],
    [clear], [dawn/dusk], [0.1873], [0.1309], [30.12%],
    [overcast], [dawn/dusk], [0.2134], [0.1380], [35.36%],
    [partly cloudy], [dawn/dusk], [0.1722], [0.1067], [38.07%],
    [snowy], [dawn/dusk], [0.2283], [0.1326], [41.92%],
    [rainy], [dawn/dusk], [0.1498], [0.0700], [53.24%],
    [snowy], [night], [0.1789], [0.0520], [70.94%],
    [rainy], [night], [0.3263], [0.1798], [44.88%],
  ),
  caption: [Mean IoU-based robustness results of S-R-V2, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.2040], [0.1455], [28.67%],
    [overcast], [daytime], [0.2155], [0.1531], [28.94%],
    [partly cloudy], [daytime], [0.2111], [0.1520], [27.98%],
    [rainy], [daytime], [0.2320], [0.1546], [33.36%],
    [snowy], [daytime], [0.2037], [0.1398], [31.38%],
    [clear], [night], [0.1457], [0.0689], [52.71%],
    [clear], [dawn/dusk], [0.2224], [0.1655], [25.58%],
    [overcast], [dawn/dusk], [0.2520], [0.1776], [29.51%],
    [partly cloudy], [dawn/dusk], [0.2163], [0.1439], [33.50%],
    [snowy], [dawn/dusk], [0.2776], [0.1808], [34.87%],
    [rainy], [dawn/dusk], [0.2011], [0.1102], [45.21%],
    [snowy], [night], [0.2315], [0.0883], [61.85%],
    [rainy], [night], [0.3691], [0.2437], [33.97%],
  ),
  caption: [Dice-based robustness results of S-R-V2, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Student - Non-Robust - V6 (S-NR-V6)]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Pixel Accuracy*], [*Adversarial Pixel Accuracy*], [*Degradation of Pixel Accuracy*]
    ),
    [clear], [daytime], [0.9038], [0.0100], [98.89%],
    [overcast], [daytime], [0.9051], [0.0079], [99.13%],
    [partly cloudy], [daytime], [0.8900], [0.0101], [98.87%],
    [rainy], [daytime], [0.9060], [0.0106], [98.83%],
    [snowy], [daytime], [0.8751], [0.0103], [98.83%],
    [clear], [night], [0.8413], [0.0044], [99.48%],
    [clear], [dawn/dusk], [0.9265], [0.0078], [99.16%],
    [overcast], [dawn/dusk], [0.9463], [0.0115], [98.79%],
    [partly cloudy], [dawn/dusk], [0.8568], [0.0030], [99.65%],
    [snowy], [dawn/dusk], [0.9467], [0.0136], [98.56%],
    [rainy], [dawn/dusk], [0.7993], [0.0117], [98.54%],
    [snowy], [night], [0.7948], [0.0005], [99.94%],
    [rainy], [night], [0.9646], [0.0056], [99.42%],
  ),
  caption: [Pixel accuracy-based robustness results of S-NR-V6, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Mean IoU*], [*Adversarial Mean IoU*], [*Degradation of Mean IoU*]
    ),
    [clear], [daytime], [0.4530], [0.0036], [99.20%],
    [overcast], [daytime], [0.4497], [0.0024], [99.48%],
    [partly cloudy], [daytime], [0.4532], [0.0030], [99.35%],
    [rainy], [daytime], [0.5187], [0.0036], [99.31%],
    [snowy], [daytime], [0.4254], [0.0033], [99.21%],
    [clear], [night], [0.3761], [0.0013], [99.65%],
    [clear], [dawn/dusk], [0.4389], [0.0025], [99.44%],
    [overcast], [dawn/dusk], [0.5675], [0.0033], [99.42%],
    [partly cloudy], [dawn/dusk], [0.3934], [0.0010], [99.76%],
    [snowy], [dawn/dusk], [0.5355], [0.0062], [98.83%],
    [rainy], [dawn/dusk], [0.3267], [0.0039], [98.80%],
    [snowy], [night], [0.2738], [0.0002], [99.92%],
    [rainy], [night], [0.6212], [0.0014], [99.77%],
  ),
  caption: [Mean IoU-based robustness results of S-NR-V6, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 5,
    table.header(
      [*Weather*], [*Time of Day*], [*Clean Dice*], [*Adversarial Dice*], [*Degradation of Dice*]
    ),
    [clear], [daytime], [0.5436], [0.0072], [98.68%],
    [overcast], [daytime], [0.5414], [0.0047], [99.14%],
    [partly cloudy], [daytime], [0.5389], [0.0058], [98.91%],
    [rainy], [daytime], [0.6125], [0.0071], [98.85%],
    [snowy], [daytime], [0.5180], [0.0066], [98.72%],
    [clear], [night], [0.4651], [0.0027], [99.43%],
    [clear], [dawn/dusk], [0.5142], [0.0049], [99.05%],
    [overcast], [dawn/dusk], [0.6537], [0.0065], [99.00%],
    [partly cloudy], [dawn/dusk], [0.4746], [0.0019], [99.60%],
    [snowy], [dawn/dusk], [0.6126], [0.0118], [98.08%],
    [rainy], [dawn/dusk], [0.4019], [0.0077], [98.08%],
    [snowy], [night], [0.3235], [0.0005], [99.86%],
    [rainy], [night], [0.6777], [0.0028], [99.59%],
  ),
  caption: [Dice-based robustness results of S-NR-V6, grouped by weather and time of day.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 2)[
  #text(size: 12pt)[Latency and Hardware Utilization]
]
#v(2 * line_spacing)

In this section, the latency and hardware utilization results will be reported. The report is split into three major parts: High-Level Latency Results, U-Net Component Results Breakdown, and Hardware Utilization.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[High-Level Latency Results]
]
#v(2 * line_spacing)

High-Level latency results present the latency without the breakdown of the U-Net component. The results are useful for the discussion of the deployment in embedded settings, allowing in-depth analysis and decision on whether or not the model is adequate for real-world deployment.

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Computational Overhead]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [9.1209], [15.9818], [8.8983], [16.1332], [16.0194], [16.1119],
    [*P50 (median)*], [10.3460], [17.7029], [10.1664], [17.8642], [17.5426], [17.7692],
    [*Average*], [10.2873], [17.7410], [10.1469], [17.9222], [17.6616], [17.7559],
    [*P75*], [11.0284], [19.0738], [10.9260], [19.2782], [18.9703], [19.1209],
    [*P90*], [12.1121], [20.1207], [12.1902], [20.4029], [20.3195], [20.2003],
    [*P95*], [12.7919], [21.0775], [12.7936], [21.2545], [21.2481], [21.1420],
    [*P99*], [13.7670], [22.5545], [13.9077], [22.7313], [25.1785], [22.3488],
    [*Max*], [15.5699], [30.6357], [16.0060], [28.7341], [37.0017], [30.0276],
  ),
  caption: [Computational overhead across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Inference Latency]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [11.3226], [11.9216], [11.3312], [11.8737], [11.3584], [11.3971],
    [*P50 (median)*], [11.3469], [12.1104], [11.3563], [12.1081], [12.0448], [12.1099],
    [*Average*], [11.3886], [11.8793], [11.4019], [11.8730], [11.8412], [11.7726],
    [*P75*], [11.3811], [12.2060], [11.3906], [12.2814], [12.1625], [12.3062],
    [*P90*], [11.4809], [12.3631], [11.4935], [12.4222], [12.4240], [12.4241],
    [*P95*], [11.7473], [12.4196], [11.7991], [12.5718], [12.8567], [12.6710],
    [*P99*], [12.0211], [12.8438], [12.2067], [12.8523], [13.1814], [12.9189],
    [*Max*], [13.7022], [14.0261], [13.4138], [13.7758], [17.1346], [14.4398],
  ),
  caption: [Inference latency across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[End-to-End Latency]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [20.4933], [27.9840], [20.2574], [28.1196], [27.6386], [28.0434],
    [*P50 (median)*], [21.7468], [29.5899], [21.5801], [29.6740], [29.3825], [29.4639],
    [*Average*], [21.6759], [29.6204], [21.5488], [29.7951], [29.5028], [29.5285],
    [*P75*], [22.4113], [31.2429], [22.3250], [31.4036], [30.9976], [31.3520],
    [*P90*], [23.4937], [32.2318], [23.6007], [32.5589], [32.5066], [32.4284],
    [*P95*], [24.1718], [33.1867], [24.2179], [33.4997], [33.2923], [33.3618],
    [*P99*], [25.1133], [34.7747], [25.3181], [34.7553], [37.2449], [34.6271],
    [*Max*], [26.9269], [42.7713], [28.4858], [40.6611], [49.8411], [42.3598],
  ),
  caption: [End-to-end latency across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[U-Net Results Breakdown]
]
#v(2 * line_spacing)

This section presents the detailed results of latency for each component inside the U-Net model. The detailed breakdown enables the discussion regarding the performance of specific parts.

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Encoder-only Latency]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [4.9125], [6.9142], [4.9098], [7.1661], [7.0287], [7.0955],
    [*P50 (median)*], [4.9290], [7.6757], [4.9276], [7.8315], [7.7393], [7.7421],
    [*Average*], [5.2335], [8.5811], [5.3011], [8.5966], [8.5096], [8.3301],
    [*P75*], [4.9619], [7.9579], [4.9606], [8.0712], [8.0545], [8.0528],
    [*P90*], [5.6288], [13.3390], [6.4799], [13.0010], [13.1279], [11.0412],
    [*P95*], [7.5153], [17.0287], [8.1460], [16.9327], [16.8453], [13.8347],
    [*P99*], [10.3834], [20.2755], [10.3439], [20.0113], [20.0247], [19.1141],
    [*Max*], [13.2217], [21.3340], [12.8358], [21.5970], [21.2020], [21.0779],
  ),
  caption: [Encoder-only latency across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Decoder-only Latency]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [5.5316], [6.1952], [5.5316], [6.1942], [6.1891], [6.1932],
    [*P50 (median)*], [5.5337], [6.2116], [5.5337], [6.2106], [6.2075], [6.2095],
    [*Average*], [5.5697], [6.2352], [5.5851], [6.2343], [6.2331], [6.2277],
    [*P75*], [5.5357], [6.2341], [5.5368], [6.2290], [6.2290], [6.2280],
    [*P90*], [5.5409], [6.3109], [5.7816], [6.2977], [6.2986], [6.2731],
    [*P95*], [5.8511], [6.3826], [5.8675], [6.3746], [6.3693], [6.3672],
    [*P99*], [6.1887], [6.7072], [6.3105], [6.7396], [6.7683], [6.5796],
    [*Max*], [7.6237], [6.9949], [7.5786], [8.3712], [8.3057], [6.9673],
  ),
  caption: [Decoder-only latency across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 4)[
  #text(size: 12pt)[Segmentation head Latency]
]
#v(2 * line_spacing)

#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*]
    ),
    [*Distilled From*], [X], [T-NR-V5], [X], [T-NR-V5], [T-R-V5], [T-R-V5],
    [*P25*], [0.5048], [0.5048], [0.5048], [0.5048], [0.5048], [0.5048],
    [*P50 (median)*], [0.5059], [0.5048], [0.5059], [0.5048], [0.5048], [0.5048],
    [*Average*], [0.5063], [0.5060], [0.5073], [0.5060], [0.5050], [0.5072],
    [*P75*], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059],
    [*P90*], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059],
    [*P95*], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059], [0.5059],
    [*P99*], [0.5069], [0.5069], [0.5069], [0.5069], [0.5059], [0.5069],
    [*Max*], [1.6568], [1.6179], [1.7746], [1.1540], [0.5110], [1.8504],
  ),
  caption: [Segmentation head latency across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing,
)

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Hardware Utilization Results]
]
#v(2 * line_spacing)

The hardware utilization results, as the name implies, present the results of hardware utilization during inference. The hardware utilization results allow discussion regarding the minimum acceptable hardware specifications suited for the model.

#v(2 * line_spacing)
#set par(
  justify: false,
  leading: line_spacing,
)
#figure(
  table(
    columns: 7,
    table.header(
      [*Model Name*], [*T-NR-V5*], [*S-NR-V5*], [*T-R-V5*], [*S-R-V1*], [*S-R-V2*], [*S-NR-V6*],
    ),
    [*Peak process CPU*], [1008.2%], [947.5%], [1006.3%], [951.8%], [1043.3%], [952.8%],
    [*Peak CPU total capacity*], [63.0%], [59.2%], [62.9%], [59.5%], [65.2%], [59.6%],
    [*Peak process RAM (MiB)*], [1273.86], [1425.39], [1273.74], [1425.45], [1425.26], [1425.41],
    [*Incremental RAM peak (MiB)*], [8.57], [31.58], [8.94], [31.46], [31.58], [31.58],
    [*Peak system RAM usage*], [12.7%], [13.4%], [12.6%], [13.4%], [13.4%], [13.40%],
    [*Peak PyTorch VRAM allocation*], [58.57], [28.16], [58.57], [28.16], [28.16], [28.16],
    [*Peak PyTorch VRAM reserve*], [184], [192], [184], [192], [192], [192],
    [*Incremental VRAM allocation*], [2.64], [2.64], [2.64], [2.64], [2.64], [2.64],
    [*Maximum CPU temperature (°C)*], [75.6], [75.6], [80.1], [78.2], [78.4], [72.8],
    [*Maximum GPU temperature (°C)*], [67], [64], [68], [66], [66], [66],
    [*Peak device VRAM (NVML)*], [803.04], [811.54], [803.04], [811.10], [811.10], [811.10],
  ),
  caption: [Hardware utilization across all model checkpoints.]
)
#set par(
  justify: true,
  leading: 2 * line_spacing
)

#pagebreak()
#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[[WIP] DISCUSSIONS]
  ]
]
#v(2 * line_spacing)


#set par(
  justify: false,
)
Research questions:
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  + How effective is the proposed adversarial distillation framework (Seg-ARD) in improving the adversarial robustness of compact student model compared to standard response-based model when subjected to multi-step PGD attacks?
  + To what extent does integrating adversarial training into knowledge distillation impact the clean segmentation fidelity across diverse driving conditions?
  + What is the impact of the proposed framework on inference latency and hardware utilization in resource-constrained deployment settings?
]

Research objectives:
#box(
  stroke: 0pt,
  inset: (left: tab_length)
)[
  + To develop and adapt an adversarial robustness distillation framework for semantic segmentation using ResNet-18-based segmentation teacher model and MobileNetV2-based segmentation student model.
  + To evaluate and benchmark the trade-off between clean segmentation performance and adversarial robustness using BDD100K dataset.
  + To quantify and analyze the inference latency and efficiency of the robust distilled model to validate its feasibility for real-time autonomous driving applications.
]

#table(
  columns: 3,
  fill: (_, y) => if y == 0 { gray.lighten(85%) },
  table.header(
    [*Research Question*], [*Specific Question*], [*Relevant Chart*],
  ),
  [RQ 1], [How much has the robustness improved?], [Comparison chart showing attack successful rate before and after adversarial training, using mean IoU.],
  [RQ 1], [How much has the robustness improved for specific driving conditions?], [Comparison chart showing attack successful rate before and after adversarial training, using mean IoU, grouped by weather and time of day.],
  [RQ 2], [What are the tradeoffs?], [Comparison chart showing the performance (pixel accuracy, mean IoU, and Dice) before and after adversarial training.],
  [RQ 2], [What are the tradeoffs for specific driving conditions?], [Comparison chart showing the performance (pixel accuracy, mean IoU, and Dice) before and after adversarial training, grouped by weather and time of day.],
  [RQ 3], [How is the latency across models?], [Chart showing the comparison of accumulated latency between models.],
  [RQ 3], [Which part of U-Net is the slowest?], [Chart showing the comparison of accumulated latency between models, grouped by each component.],
  [RQ 3], [How is the hardware utilization across models?], [Hardware utilization table.],
)

#pagebreak()
#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[[WIP] CONCLUSIONS AND FUTURE WORK]
  ]
]
#v(2 * line_spacing)

#pagebreak()
#align(center)[
  #heading(level: 1, numbering: none)[
    #text(size: 12pt)[REFERENCES]
  ]
]
#v(2 * line_spacing)

#set par(
  first-line-indent: (amount: 0cm, all: true),
  hanging-indent: tab_length,
  justify: true
)

Abdoli, N., Sadri, Y., Gilley, P., Zhang, K., Chen, Y., & Qiu, Y. (2025). From CNNs to SAM: A survey of Deep learning techniques for liver tumor segmentation in CT images. _IEEE Access_, _13_, 199074-199104. https://doi.org/10.1109/access.2025.3631322

Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., & Zou, J. (2019). Gradio: Hassle-Free sharing and testing of ML models in the Wild. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.1906.02569

Acharya, K., Velasquez, A., & Song, H. H. (2024). A survey on Symbolic Knowledge Distillation of large Language models. _IEEE Transactions on Artificial Intelligence_, _5_(12), 5928-5948. https://doi.org/10.1109/tai.2024.3428519

Arnab, A., Miksik, O., & H. S. Torr, P. (2020). On the robustness of semantic segmentation models to adversarial attacks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, _42_(12), 3040-3053. https://doi.org/10.1109/TPAMI.2019.2919707

Ayas, M. S., Ayas, S., & Djouadi, S. M. (2022). Projected gradient descent adversarial attack and its defense on a fault diagnosis system. In _2022 45th International Conference on Telecommunications and Signal Processing (TSP)_ (pp. 36-39). https://doi.org/10.1109/tsp55681.2022.9851334

Bachani, V., Roychowdhury, A., Dasgupta, A., & El-Alfy, H. (2024). Image Segmentation Survey: Classical and Deep learning methods. In _2024 International Conference on Electrical, Computer and Energy Technologies (ICECET)_ (pp. 1-6). https://doi.org/10.1109/icecet61485.2024.10698602

Buciluǎ, C., Caruana, R., & Niculescu-Mizil, A. (2006, August). Model compression. In _Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining_ (pp. 535-541).

Brar, K. K., Goyal, B., Dogra, A., Mustafa, M. A., Majumdar, R., Alkhayyat, A., & Kukreja, V. (2024). Image segmentation review: Theoretical background and recent advances. _Information Fusion_, _114_, 102608. https://doi.org/10.1016/j.inffus.2024.102608

Chariton, A. (2023, January 1). _Knowledge distillation tutorial_. Retrieved August 13, 2026, from https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html

Chen, M., Wang, F., Wang, W., Gao, X., Liu, Z., & Cao, S. (2026). LG-RobustSAM: A lightweight and robust segmentation model for degraded scenarios. _Neurocomputing_, 683, 133556. https://doi.org/10.1016/j.neucom.2026.133556

Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. _Lecture Notes in Computer Science_, _9901_, 424-432. https://doi.org/10.1007/978-3-319-46723-8_49

Corke, P. (2023). Robotics, vision and control. In _Springer tracts in advanced robotics_. https://doi.org/10.1007/978-3-031-06469-2

Dai, J., He, K., & Sun, J. (2016). Instance-Aware Semantic Segmentation via Multi-task Network Cascades. In _2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ (pp. 3150-3158). IEEE. https://doi.org/10.1109/cvpr.2016.343

Fang, Z., & Yang, Y. (2023). Knowledge distillation across vision and language. In _Studies in computational intelligence_ (pp. 65-94). https://doi.org/10.1007/978-3-031-32095-8_3

Goldblum, M., Fowl, L., Feizi, S., & Goldstein, T. (2020). Adversarially robust distillation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 34, pp. 3996-4003). https://aaai.org/ojs/index.php/AAAI/article/view/5816/5672

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.1412.6572

Gross, S., Tong, X., Chanan, G., Van Der Maaten, L., Trottier, L., & Sokolov, D. (2019). _Fb.resnet.torch_ [Software]. https://github.com/facebookarchive/fb.resnet.torch

Gu, W., Bai, S., & Kong, L. (2022). A review on 2D instance segmentation based on deep neural networks. _Image and Vision Computing_, _120_, 104401. https://doi.org/10.1016/j.imavis.2022.104401

Hall, P., Curtis, J., & Pandey, P. (2023). _Machine Learning for High-Risk Applications: Techniques for Responsible AI_. O'Reilly Media.

Han, S., Lin, Y., Guo, Z., & Lv, K. (2022). A lightweight and style-robust neural network for autonomous driving in end side devices. _Connection Science_, _35_(1). https://doi.org/10.1080/09540091.2022.2155613

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.1512.03385

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.1503.02531

Kargin, A., & Petrenko, T. (2023). Knowledge distillation for autonomous intelligent unmanned system. In _Studies in computational intelligence_ (pp. 193-230). https://doi.org/10.1007/978-3-031-32095-8_7

Khamaiseh, S. Y., Bagagem, D., Al-Alaj, A., Mancino, M., & Alomari, H. W. (2022). Adversarial Deep Learning: A survey on adversarial attacks and defense mechanisms on image classification. _IEEE Access_, _10_, 102266-102291. https://doi.org/10.1109/access.2022.3208131

Lakshmanan, V., Görner, M., & Gillard, R. (2021). _Practical machine learning for computer vision: End-To-End Machine Learning for Images_. O'Reilly Media.

Lee, J., Yoon, J., Kim, J., & Choi, J. W. (2026). Image-text driven style randomization for domain generalized semantic segmentation. _Neurocomputing_, 676, 132953. https://doi.org/10.1016/j.neucom.2026.132953

Lê, M. T., Wolinski, P., & Arbel, J. (2026). Efficient Neural Networks for tiny Machine Learning: A Comprehensive Review. _ACM Transactions on Intelligent Systems and Technology_, _17_(4), 1-41. https://doi.org/10.1145/3798276

Li, G., Togo, R., Ogawa, T., & Haseyama, M. (2022). Self-Knowledge Distillation based Self-Supervised Learning for Covid-19 Detection from Chest X-Ray Images. _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 1371-1375. https://doi.org/10.1109/icassp43922.2022.9746540

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In _2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ (pp. 3431-3440). https://doi.org/10.1109/cvpr.2015.7298965

Lunghi, D., Simitsis, A., Caelen, O., & Bontempi, G. (2023). Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives. In _DEC '23: Proceedings of the Second ACM Data Economy Workshop_ (pp. 27'33). https://doi.org/10.1145/3600046.3600051

Ma, J., Chen, J., Ng, M., Huang, R., Li, Y., Li, C., Yang, X., & Martel, A. L. (2021). Loss odyssey in medical image segmentation. _Medical Image Analysis_, _71_, 102035. https://doi.org/10.1016/j.media.2021.102035

Minaee, S., Boykov, Y. Y., Porikli, F., Plaza, A. J., Kehtarnavaz, N., & Terzopoulos, D. (2021). Image segmentation using Deep Learning: A survey. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, _44_(7), 1. https://doi.org/10.1109/tpami.2021.3059968

Moawad, A., Zebiak, M., Han, J., Karbowski, D., Zhang, Y., & Rousseau, A. (2024). Effect of adaptive cruise control on fuel consumption in real-world driving conditions. _Nature Communications_, _15_(1), 10016. https://doi.org/10.1038/s41467-024-54066-8

Moslemi, A., Briskina, A., Dang, Z., & Li, J. (2024). A survey on knowledge distillation: Recent advancements. _Machine Learning With Applications_, _18_, 100605. https://doi.org/10.1016/j.mlwa.2024.100605

Motorist. (2024, September 22). _How cruise control makes long drives a breeze_. Motorist.ph. Retrieved July 29, 2026, from https://www.motorist.ph/article/3544/how-cruise-control-makes-long-drives-a-breeze

Muhammad, A., & Bae, S. (2022). A survey on Efficient Methods for Adversarial Robustness. _IEEE Access_, 10, 118815-118830. https://doi.org/10.1109/access.2022.3216291

Noh, H., Hong, S., & Han, B. (2015). Learning Deconvolution Network for Semantic Segmentation. In _ICCV '15: Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV)_ (pp. 1520-1528). https://doi.org/10.1109/iccv.2015.178

Park, S., Rakha, H., Ahn, K., & Moran, K. (2013). Fuel economy impacts of manual, conventional cruise control, and predictive Eco-Cruise control driving. _International Journal of Transportation Science and Technology_, _2_(3), 227-242. https://doi.org/10.1260/2046-0430.2.3.227

Qiao, Y., Sathyanarayana, N. B., Shi, C., He, Z., Wang, T., & Hou, T. (2026). A survey on adversarial machine learning: Attacks, defenses, real-world applications, and future research directions. _Neurocomputing_, _671_, 132670. https://doi.org/10.1016/j.neucom.2026.132670

Rayed, M. E., Islam, S. S., Niha, S. I., Jim, J. R., Kabir, M. M., & Mridha, M. (2024). Deep learning for medical image segmentation: State-of-the-art advancements and challenges. _Informatics in Medicine Unlocked_, 47, 101504. https://doi.org/10.1016/j.imu.2024.101504

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. _Lecture Notes in Computer Science_, _9351_, 234-241. https://doi.org/10.1007/978-3-319-24574-4_28

Rossolini, G., Nesti, F., D'Amico, G., Nair, S., Biondi, A., & Buttazzo, G. (2023). On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving. _IEEE Transactions on Neural Networks and Learning Systems_, _35_(12), 18328-18342. https://doi.org/10.1109/tnnls.2023.3314512

Saeedizadeh, N., Jalali, S. M. J., Khan, B., Kebria, P. M., & Mohamed, S. (2024). A new optimization approach based on neural architecture search to enhance deep U-Net for efficient road segmentation. _Knowledge-Based Systems_, _296_, 111966. https://doi.org/10.1016/j.knosys.2024.111966

Seghier, M. L. (2024). Image Segmentation Evaluation with the DICE Index: Methodological issues. _International Journal of Imaging Systems and Technology_, _34_(6). https://doi.org/10.1002/ima.23203

Shafahi, A., Najibi, M., Ghiasi, A., Xu, Z., Dickerson, J., Studer, C., S. Davis, L., Taylor, G., & Goldstein, T. (2019). Adversarial training for free! In _Proceedings of the 33rd International Conference on Neural Information Processing Systems_ (pp. 3358-3369). Curran Associates Inc. https://dl.acm.org/doi/10.5555/3454287.3454589

Shang, Y., Li, D., & Wang, Y. (2025). A survey on Adversarial Attack and Defense in image Classification. In _AICCC '25: Proceedings of the 2025 8th Artificial Intelligence and Cloud Computing Conference_ (pp. 363-371). https://doi.org/10.1145/3789982.3790029

Song, Y., Zhang, P., Huang, W., Zha, Y., & Zhang, Y. (2025). Flexible temperature parallel distillation for dense object detection: Make Response-Based Knowledge distillation great again. _IEEE Transactions on Circuits and Systems for Video Technology_, _35_(5), 4963-4975. https://doi.org/10.1109/tcsvt.2024.3525051

Stapel, J., Mullakkal-Babu, F. A., & Happee, R. (2018). Automated driving reduces perceived workload, but monitoring causes higher cognitive load than manual driving. _Transportation Research Part F Traffic Psychology and Behaviour_, _60_, 590-605. https://doi.org/10.1016/j.trf.2018.11.006

Xiao, Z., & Tong, H. (2025). Federated Contrastive Learning with Feature-Based distillation for human activity recognition. _IEEE Transactions on Computational Social Systems_, _12_(4), 1759-1772. https://doi.org/10.1109/tcss.2024.3510428

Xie, C., Wang, J., Zhang, Z., Zhou, Y., Xie, L., & Yuille, A. (2017). Adversarial examples for semantic segmentation and object detection. In _2017 IEEE International Conference on Computer Vision (ICCV)_ (pp. 1378-1387). IEEE. https://doi.org/10.1109/iccv.2017.153

Xie, Y., Shi, Y., Xie, C., Hu, Q., Liu, Y., & Lin, C. (2025). A knowledge distillation-based object detection model focused on road scene perception and localization. _Neurocomputing_, _656_, 131501. https://doi.org/10.1016/j.neucom.2025.131501

Yang, C., Yu, X., An, Z., & Xu, Y. (2023). Categories of Response-Based, Feature-Based, and Relation-Based knowledge distillation. In _Studies in computational intelligence_ (pp. 1-32). https://doi.org/10.1007/978-3-031-32095-8_1

Yao, S., Guan, R., Huang, X., Li, Z., Sha, X., Yue, Y., Lim, E. G., Seo, H., Man, K. L., Zhu, X., & Yue, Y. (2023). Radar-Camera Fusion for object detection and semantic Segmentation in Autonomous Driving: A Comprehensive Review. _IEEE Transactions on Intelligent Vehicles_, _9_(1), 2094-2128. https://doi.org/10.1109/tiv.2023.3307157

Yerlikaya, F. A., & Bahtiyar, Ş. (2026). A survey of adversarial attacks on machine learning. _Neurocomputing_, _670_, 132573. https://doi.org/10.1016/j.neucom.2025.132573

Yin, H., Wang, R., Liu, B., & Yan, J. (2022). On adversarial robustness of semantic segmentation models for automated driving. _2022 IEEE Intelligent Vehicles Symposium (IV)_, 867-873. https://doi.org/10.1109/iv51971.2022.9827460

Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2018). BDD100K: a diverse driving dataset for heterogeneous multitask learning. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.1805.04687

Yu, R., Zhang, X., Zhao, R., Yan, H., & Wang, M. (2025). DistillDrive: End-to-End Multi-Mode autonomous driving distillation by isomorphic Hetero-Source Planning Model. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_. https://openaccess.thecvf.com/content/ICCV2025/html/Yu_DistillDrive_End-to-End_Multi-Mode_Autonomous_Driving_Distillation_by_Isomorphic_Hetero-Source_Planning_ICCV_2025_paper.html

Zhang, H., Yu, Y., Jiao, J., P. Xing, E., Ghaoui, L. E., & Jordan, M. (2019). Theoretically principled trade-off between robustness and accuracy. In K. Chaudhuri & R. Salakhutdinov (Eds.), _Proceedings of the 36th International Conference on Machine Learning_ (Vol. 97, pp. 7472-7482). PMLR. https://proceedings.mlr.press/v97/zhang19p.html

Zhang, Z., Liu, Q., & Wang, Y. (2018). _Road extraction by deep residual U-Net. IEEE Geoscience and Remote Sensing Letters_, _15_(5), 749-753. https://doi.org/10.1109/lgrs.2018.2802944

Zhao, Y., Xu, X., Li, R., Song, R., & Zhang, W. (2025). RAKD: Risk-Aware Knowledge Distillation for Autonomous Driving in Adversarial Scenarios. _IEEE Transactions on Vehicular Technology_, _75_(3), 3845-3858. https://doi.org/10.1109/tvt.2025.3613697
