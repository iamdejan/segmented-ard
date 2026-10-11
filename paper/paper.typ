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
  + What is the impact of the proposed framework on inference latency and computational overhead in resource-constrained deployment settings?
]

// TODO: decide if we want to keep RQ2 unchanged, or adjust to ensure narrower scope. Currently, in the evaluation scripts, we have yet to measure across diverse driving conditions.

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
  #text(size: 12pt)[Resesarch Scope and Contribution]
]
#v(2 * line_spacing)

This research will cover the development and experiments on . As part for this research, multiple adversarial attack techniques will be used to test the model's resistance against each type of the attack.

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
  #text(size: 12pt)[Models Used and Knowledge Distillation Procedure]
]
#v(2 * line_spacing)

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

Since this study focuses on semantic segmentation, there should be an adjustment to the formula. The adjusted formula is:
$
  text("ASR")_(text("untargeted")) = 1 - (text("mIoU")_(text("adversarial"))) / (text("mIoU")_(text("clean")))
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

Lower ASR is preferrable, signaling high robustnesss that the model possesses.

#v(2 * line_spacing)
#heading(level: 3)[
  #text(size: 12pt)[Latency]
]
#v(2 * line_spacing)

Latency is defined as the total time taken between submitting the input image and receiving the output. In real-time applications, especially in cruise control and self-driving, latency is important because the segmented output is used to assist the driving capability. Higher latency means longer duration needed for the vehicle to take decision, which increases the risk of accident. Therefore, this study aims to lower the latency.

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
