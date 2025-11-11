# Beyond Leaf and Word: A Multimodal Analysis of Nahua Botanical Knowledge in the De la Cruz-Badiano Herbal

**Summary**  
This study investigates the structure of Nahua botanical knowledge as captured in the 1552 De la Cruz-Badiano Herbal, a unique document preserving Indigenous plant names and illustrations. Our goal was to determine if and how different data modalities—linguistic, morphological, and structural—align in a shared conceptual space, providing quantitative insight into this understudied worldview.

We employed a multi-modal approach, first creating a dataset with three aligned modalities: leaf shape morphometrics from hand-traced illustrations, text embeddings of corresponding English and Spanish texts, and node embeddings of a graph representing Nahuatl name co-occurrence. A three-tower CNN with a contrastive loss function was then trained to learn a common embedding space that minimized the distance between these modalities. Synthetic leaf data was used to augment the training set.

The model successfully learned a common embedding space for all three modalities. We found that the leaf shape data occupied a more distinct region of this space, while the linguistic and structural data were closely aligned. This suggests that the cultural and functional associations embedded in the Nahuatl names and texts are more strongly correlated with each other than with a plant's physical shape.

Our results indicate a sophisticated Indigenous classification system where the structure of language and name relationships provides a primary framework for understanding plants, distinct from a purely morphological basis. This research highlights the value of using computational methods to re-examine historical documents, offering a path to decolonize plant science by quantifying the richness of Indigenous botanical knowledge.

**Keywords**  
Botany, De la Cruz-Badiano Herbal, Indigenous knowledge, Libellus de Medicinalibus Indorum Herbis, Multimodal learning, Nahuatl, Plant classification

## Figures and tables  
![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_coincidence_graph.png)  
**Figure 1: Coincidence graph of Nahuatl names.** Vertices representing Nahuatl names connected by weighted edges based on coincidence in subchapters. Vertices are colored by type: green, plant; magenta, stone; lavender, bird; orange, animal. Only nodes with ten or more connections are shown for visualization purposes.  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_combined_clustering_and_wordclouds_fasttext_umap_tficf.png)  
**Figure 2: Clusters based on text embeddings.** UMAP of FastText embeddings in A) English and B) Spanish and word frequency clouds for five clusters (indciated by color).  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/leaf_traces_1.png)  
**Figure 3: Leaf traces projected on original illustrations.** Each leaf trace is indicated by color and plotted on top of the original image. Continued in the next figure.  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/leaf_traces_2.png)  
**Figure 4: Leaf traces projected on original illustrations.** Each leaf trace is indicated by color and plotted on top of the original image. Continued from the previous figure.  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_ECT.png)  
**Figure 5: Radial Euler Characteristic Transform (ECT).** For random leaves from each indicated Nahuatl plant, the radial ECT aligned with the corresponding leaf shape outline. The radial ECT and shape mask for the leaf are the inputs for a 2-channel Convolutional Neural Network (CNN).  

| Class/Average   |   Precision (Leaves) |   Recall (Leaves) |   F1 (Leaves) |   Precision (English) |   Recall (English) |   F1 (English) |   Precision (Spanish) |   Recall (Spanish) |   F1 (Spanish) |
|:----------------|---------------------:|------------------:|--------------:|----------------------:|-------------------:|---------------:|----------------------:|-------------------:|---------------:|
| Xihuitl         |                 0.99 |              0.66 |          0.79 |                  0.77 |               0.55 |           0.64 |                  0.78 |               0.68 |           0.72 |
| Xochitl         |                 0.99 |              0.9  |          0.94 |                  0.7  |               0.62 |           0.65 |                  0.73 |               0.62 |           0.67 |
| Quahuitl        |                 0.99 |              0.94 |          0.96 |                  0.6  |               0.8  |           0.69 |                  0.52 |               0.93 |           0.67 |
| Patli           |                 0.98 |              0.62 |          0.76 |                  1    |               0.35 |           0.52 |                  0.56 |               0.59 |           0.57 |
| Quilitl         |                 0.99 |              0.74 |          0.85 |                  0.44 |               1    |           0.62 |                  0.44 |               1    |           0.62 |
| micro avg       |                 0.99 |              0.72 |          0.83 |                  0.66 |               0.61 |           0.63 |                  0.62 |               0.71 |           0.66 |
| macro avg       |                 0.99 |              0.77 |          0.86 |                  0.7  |               0.66 |           0.62 |                  0.6  |               0.76 |           0.65 |
| weighted avg    |                 0.99 |              0.72 |          0.83 |                  0.74 |               0.61 |           0.63 |                  0.66 |               0.71 |           0.66 |

**Table 1:** Unimodal CNN performance for leaf shapes and English and Spanish text embeddings for the five indicated Nahuatl plant categories.

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_CNN_morpheme.png) 
**Figure 6: Shape features and text distingishing Nahuatl plant groupings.** The three Nahuatl botanical classes (Xihuitl, Xochitl, Quahuitl, Patli, and Quilitl) by row and, left to right, Gradient-weighted Class Activation Mapping (GradCAM) and English and Spanish word clouds that characterize each class by column.  

| Metric                      | Image-to-Text | Text-to-Image | Image-to-Graph | Graph-to-Image | Text-to-Graph | Graph-to-Text |
| :-------------------------- | ------------: | ------------: | -------------: | -------------: | ------------: | ------------: |
| MAP avg.                    |        0.4034 |        0.3624 |         0.4628 |         0.4144 |        0.9288 |         0.925 |
| MAP std.                    |        0.0178 |        0.0214 |         0.0248 |           0.03 |        0.0065 |        0.0075 |
| Recall@1 score avg.         |        0.2761 |        0.5964 |         0.3093 |         0.6725 |        0.8686 |        0.9272 |
| Recall@1 score std.         |        0.0189 |        0.0848 |         0.0308 |         0.0709 |        0.0236 |        0.0019 |
| Recall@1 above chance avg.  |        9.8453 |       21.2748 |        11.0305 |        23.9825 |       30.9696 |       33.0591 |
| Recall@5 score avg.         |        0.2992 |         0.894 |         0.3501 |         0.9567 |         0.894 |         0.938 |
| Recall@5 score std.         |        0.0217 |        0.0399 |         0.0276 |         0.0151 |        0.0193 |        0.0024 |
| Recall@5 above chance avg.  |        2.3761 |        7.0972 |         2.7806 |         7.5963 |        7.0975 |        7.4475 |
| Recall@10 score avg.        |        0.3428 |        0.9503 |         0.3939 |         0.9897 |        0.9336 |        0.9663 |
| Recall@10 score std.        |        0.0203 |        0.0088 |         0.0259 |         0.0031 |        0.0328 |        0.0009 |
| Recall@10 above chance avg. |        1.5383 |        4.2645 |          1.768 |         4.4411 |         4.189 |        4.3362 |

**Table 2:** Multimodal Three Tower CNN performance for each pair and direction of modalities.  

![Alt](https://github.com/DanChitwood/DeLaCruzBadiano1552/blob/main/analysis/outputs/figures/fig_three_towers.png)  
**Figure 7: A common embedding space for image, text, and graph modalities.** A) Text, B) image, and C) graph embeddings each highlighted in turn in a UMAP of a common embedding space. D) For each pair of modalities, a histogram showing Euclidean distance between pairs in the UMAP projection.  

## Code and data  
All scripts to reproduce these analyses are found in `./analysis/scripts/`. Data to reproduce these analyses are found in `./analysis/data/`. The folders `./analysis/data/leaf_traces/`, `./analysis/data/texts/`, and `./analysis/data/texts_ES/` (leaf trace data and English and Spanish texts, respectively) will need to be unzipped before using. The FastText English and Spanish models `cc.en.300.bin` and `cc.es.300.bin` will need to be downloaded from this [link](https://fasttext.cc/docs/en/crawl-vectors.html) and placed into `./analysis/data/`. The pages of the document from which leaf traces are derived and on which they are plotted are derived from the public version of [The De la Cruz-Badiano Aztec Herbal of 1552](https://archive.org/details/aztec-herbal-of-1552) and the exact images found on [figshare](https://doi.org/10.6084/m9.figshare.29825189.v1) should be downloaded, the .zip file decompressed, and saved as the folder `pics` in `./analysis/data/leaf_traces/`.

## Methods
The objective of this study was to analyze and classify Nahuatl plant names by integrating three distinct data modalities: linguistic, morphological, and structural. The data was derived from the 1939 William Gates English translation of the de la Cruz-Badiano Nahuatl Herbal. The pipeline consisted of unimodal analyses to characterize each data source, followed by a multi-modal approach to create a unified representation.

### Data Acquisition and Preprocessing
The primary data sources included digitized illustrations, corresponding English text descriptions, and a structural representation of plant name co-occurrence. For the morphological data, over 4,000 leaf outlines were hand-traced from illustrations using Fiji/ImageJ. These tracings were aligned using Generalized Procrustes Analysis (GPA) to normalize for position, scale, and rotation. The aligned shapes were then represented as a two-channel input for a convolutional neural network (CNN), with one channel being a radial Euler Characteristic Transform (ECT) in polar coordinates and the other a corresponding shape mask.

For the linguistic data, the English text from each plant's subchapter was aggregated. An automated Spanish translation was also generated for parallel analysis. FastText was employed to create word embeddings for both the English and Spanish texts. These embeddings were used for subsequent classification and clustering analyses.

A graph-based structural modality was created by constructing a network of Nahuatl plant names. Vertices in this graph represented unique Nahuatl names, and edges were weighted by the number of times two names co-occurred within the same subchapter. Node embeddings for this graph were then generated using the Node2Vec algorithm, with an embedding size of 128, a walk length of 30, and 200 walks per node.

### Data Augmentation
To address class imbalance and increase sample size for model training, a synthetic data generation method was used. After a Principal Component Analysis (PCA) was performed on the hand-traced leaf shapes, a Synthetic Minority Over-sampling Technique-like (SMOTE-like) approach was applied. Synthetic leaf shapes were created by sampling neighbors within the same class in the PCA space and linearly interpolating new shapes. These synthetic leaves were used to augment the training data for the image-based CNNs.

### Unimodal Analysis and Classification
Separate CNNs were trained to classify each modality into one of five main Nahuatl categories: the three main botanical categories (Xochitl, Quahuitl, Xihuitl) and two economic categories (Patli, Quilitl).

The image-based CNN used a two-channel input of radial ECT and shape mask to classify leaves. It was trained using stratified 5-fold cross-validation and employed ensemble prediction on the logits. To interpret the model's decisions, Grad-CAM visualizations were generated to highlight the image regions most relevant to classification.

A similar CNN architecture was adapted to classify the text data. This model took the FastText embeddings as input and was trained to classify the corresponding Nahuatl plant names. The key features distinguishing the classes were visualized using word clouds, which highlighted the most frequent and unique terms associated with each category.

### Multimodal Analysis and Integration
A three-tower CNN was developed to create a unified embedding space where the representations of all three modalities for the same plant were aligned. The model consisted of three parallel towers: an image tower (a 2D CNN), a text tower (a 1D CNN), and a graph tower (a linear layer). Each tower projected its modality's data into a shared 128-dimensional embedding space. The training objective was to minimize the distance between the embeddings of corresponding samples from all three modalities.

The model was trained using a combination of real and synthetic data within a stratified 5-fold cross-validation framework. A custom MultiModal Contrastive Loss function was used to simultaneously bring together the embeddings of the same plant across different modalities while pushing apart those of different plants.

The performance of this retrieval-focused model was evaluated using metrics such as Recall@k and Mean Average Precision (mAP) for all six possible retrieval directions (e.g., image-to-text, graph-to-image). To visualize the learned common embedding space, Uniform Manifold Approximation and Projection (UMAP) was applied to reduce the embeddings to two dimensions. A final figure was generated to illustrate the clustering of the different modalities in this space, and a histogram quantified the pairwise Euclidean distances between them.
