## Meta-GTNRP: a novel few-shot GNN-Transformer approach for nuclear receptor binding activity prediction

### Correction of Minor Error: In the published work (https://doi.org/10.1186/s13321-024-00902-4), there is a minor phrasing error in the text. The phrase "has promoted to a great deal of research" should be corrected to "has promoted a great deal of research". This correction does not affect the overall findings or conclusions of the study, but I wanted to clarify it for accuracy.


In this paper, we propose a few-shot GNN-Transformer, Meta-GTNRP to model the local and global information of molecular graph embeddings using a two-module meta-learning framework for NR-binding activity prediction. This few-shot learning strategy combines the information of 11 individual predictive tasks for 11 different NRs in a joint learning procedure to predict the binding, agonist and antagonist activity with just a few labeled compounds in highly imbalanced scenarios.

First, a GIN module treats molecules as a set of node and edge features converted into graph embeddings by neighborhood aggregation. Then, a Transformer module preserves the global information in these vectorial embeddings to propagate deep representations across attention layers. The long-range dependencies captured express the global-semantic structure of molecular embeddings as a function to predict the NR-binding properties in small drug repositories.

![ScreenShot](figures/meta-gtnrp.png?raw=true)

To address the challenge of low-data learning, we introduce a two-module meta-learning framework to quickly update model parameters across few-shot tasks for 10 different NRs to predict the binding activity of compounds on 1 new NR in the test data.

![ScreenShot](figures/meta.png?raw=true)

Experiments with a compound database (NURA dataset) containing annotations on the molecular activity on 11 different NRs demonstrate the superior performance of Meta-GTNRP over standard graph-based methods (GIN, GCN, GraphSAGE).

This repository provides the source code and datasets for the proposed work.

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Correction of Minor Error:

In the published work (https://doi.org/10.1186/s13321-024-00902-4), there is a minor phrasing error in the text. The phrase "has promoted to a great deal of research" should be corrected to "has led to a great deal of research."

This correction does not affect the overall findings or conclusions of the study, but I wanted to clarify it for accuracy.

## Data Availability and Pre-Processing

In this work, data is collected using a public compound repository known as [NURA (NUclear Receptor Activity) dataset (Valsecchi et al.) (2020)](https://www.sciencedirect.com/science/article/pii/S0041008X20303707?via=ihub), which includes publicly available information on the activity of 15247 compounds on 11 human NRs.

Data is pre-processed and transformed into molecular graphs using RDKit.Chem. 

Data pre-processing, pre-trained and GNN models are implemented based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Python Packages

We used the following Python packages for core development. We tested on Python 3.9.

```
- torch = 1.13.0+cu116 
- torch-cluster = 1.6.1
- torch-geometric =  2.3.0
- torch-scatter = 2.1.1
- torch-sparse = 0.6.17
- torch-spline-conv =  1.2.2
- torchvision = 0.14.0
- scikit-learn = 1.2.2
- seaborn = 0.12.2
- scipy = 1.7.3
- numpy = 1.22.3
- tqdm = 4.65.0
- tsnecuda = 3.0.1
- tqdm = 4.65.0
- matplotlib = 3.4.3 
- pandas = 1.5.3 
- networkx =  3.1.0
- rdkit = 2022.03.5

```

## References

[1] Valsecchi, C., Grisoni, F., Motta, S., Bonati, L., & Ballabio, D. (2020). NURA: A curated dataset of nuclear receptor modulators. Toxicology and Applied Pharmacology, 407. https://doi.org/10.1016/j.taap.2020.115244

```
@article{valsecchi2020nura,
   author = {Cecile Valsecchi and Francesca Grisoni and Stefano Motta and Laura Bonati and Davide Ballabio},
   doi = {10.1016/j.taap.2020.115244},
   journal = {Toxicology and Applied Pharmacology},
   title = {NURA: A curated dataset of nuclear receptor modulators},
   year = {2020}
}
```

[2] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.: Strategies for pre-training graph neural networks. CoRR abs/1905.12265 (2020). https://doi.org/10.48550/ARXIV.1905.12265

```
@inproceedings{hu2020pretraining,
  title={Strategies for Pre-training Graph Neural Networks},
  author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=HJlWWJSFDH}
}
```

[3] Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: 34th International Conference on Machine Learning, ICML 2017, vol. 3 (2017). https://doi.org/10.48550/arXiv.1703.03400

```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}

```

[4] Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., & Chawla, N. V. (2021). Few-shot graph learning for molecular property prediction. In The Web Conference 2021 - Proceedings of the World Wide Web Conference, WWW 2021 (pp. 2559–2567). Association for Computing Machinery, Inc. https://doi.org/10.1145/3442381.3450112
```
@article{guo2021few,
  title={Few-Shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2102.07916},
  year={2021}
}
```

[5] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. https://doi.org/10.48550/arxiv.2010.11929

```
@article{Dosovitskiy2020,
   author = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
   doi = {10.48550/arxiv.2010.11929},
   month = {10},
   title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
   url = {https://arxiv.org/abs/2010.11929},
   year = {2020},
}
```

[6] Vision Transformers with PyTorch. https://github.com/lucidrains/vit-pytorch

```
@misc{Phil Wang,
  author = {Phil Wang},
  title = {Vision Transformers},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lucidrains/vit-pytorch}}
}
```

[7] Cortes-Ciriano, I. (2016). Bioalerts: A python library for the derivation of structural alerts from bioactivity and toxicity data sets. Journal of Cheminformatics, 8(1). https://doi.org/10.1186/s13321-016-0125-7

```
@article{bioalerts,
   author = {Isidro Cortes-Ciriano},
   doi = {10.1186/s13321-016-0125-7},
   issn = {17582946},
   issue = {1},
   journal = {Journal of Cheminformatics},
   keywords = {Bioactivity,Circular fingerprints,Morgan fingerprints,Structural alerts,Toxicology},
   month = {3},
   publisher = {BioMed Central Ltd.},
   title = {Bioalerts: A python library for the derivation of structural alerts from bioactivity and toxicity data sets},
   volume = {8},
   year = {2016},
}
```



