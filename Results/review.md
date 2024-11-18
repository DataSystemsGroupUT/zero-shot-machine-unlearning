Dear Reviewers, thank you so much for your feedback. In the following paragraphs, we will address the main concerns raised and clarify our points of view. 

This paper was submitted as a short research paper that could explain the limited number of experiments presented. However, based on the relevance of the comments received, we decided to address those gaps by uploading more results to the GitHub repository associated with this project [add link here]. 

First, regarding the comments addressing the design of the experiments, we evaluated the effectiveness of our technique using the same methodology utilized by the other papers addressing the same challenge. For example, EMMN and GTK [4], both, used AllCNN and VGG16 along with the same datasets we used. Further, most non-zero-shot tecshniques such as SCRUB [5] and UNSIR [7], indeed, used the same evaluation setup. However, we decided to extend our experiments by addressing an additional dataset and an additional model; namely, CIFAR100, and ResNet. Further, the “lack the natural family of baselines, which (a) generates a dataset from model inversion and (b) applies existing methods for unlearning ”, as pointed out by Reviewer 2, is because, to the best of our knowledge, we are the only work utilizing this technique. 

Second, regarding the comments addressing the lack of an ablation study, we clearly see how our study would benefit from it. For our proposed technique, the main hyper-parameters are the dataset size and the learning rate’s magnitude. Indeed, depending on the dataset and the model used, those hyper-parameters needed special tuning. We attached a details description of the hyper-parameter selection to the GitHub repository. Further, we presented an ablation study explaining the effect of the dataset size and the learning rate on the effectiveness of unlearning.

Third, regarding the comments addressing the scalability of our proposed technique, we would like to point out the following: Our system indeed is computationally expensive as pointed out by the execution time experiments (refer to Table 7). However, it is designed to address the use case of not having access to any of the retained or forgotten datasets. In such cases, our system could be effectively used for 1) batch unlearning or 2) background async unlearning. Throughout our experimentation, model inversion proved to be very effective with most models and on various complex datasets. The technique proposed here used the simplest flavor of model inversion where we assumed no knowledge of the model or the data used for training. However, by knowing the domain of the data, i.e. face or skin tumor images, style-GANs can be effectively used to extract realistic samples representing the target class. 

Fourth, regarding the comments addressing the relevance of this paper to the conference, we find it highly relevant for the following reasons: Currently, most data management applications include predictive, ML- based components. According to GDPR, applying the “right-to-be-forgotten” would necessitate the usage of machine unlearning techniques. Variety of papers addressing the similar tasks have been published here. For example, [2], [8], [6], and [1] are related to private data processing. Further, [3] discusses the GDPR compliance framework that includes “strong” data erasure procedures, i.e. machine unlearning. 

## Comments on the newly attached tables:

In Tables A and B, the forgetting effectiveness increases as we increase the data points. However, after a certain threshold, depending on the model and the dataset, the model experiences over-forgetting, which is reflected in the decreasing Dr accuracy and the increasing MIA score. It is worth mentioning that zMuGAN doesn’t offer the same level of control over the number of samples generated per class. This is mainly due to the stochastic nature of model inversion. In our work, an ensemble of generators was utilized to increase the diversity of the generated output. 

In Tables C and D, we listed the impact of LR on the performance of zMuGAN. The results show a similar pattern. Having a lower learning rate would reduce the effect of the unlearning algorithm; on the other hand, having a learning rate higher than needed would have either a diminishing or negative impact. As requested by some of the reviewers, table E shows the performance of our method using more complex models, namely, a ResNet9 model trained on the SVHN dataset. Our technique, ZMUGAN, performs better than EMMN. and has a performance close to JiT which requires access to the forget dataset. Further, in Table F, we demonstrate the performance of our technique to forget a class out of the 100 classes represented in the CIFAR100 dataset. Our technique performs similarly to UNSIR and JiT techniques. Also, our technique scores the best MIA score, 0.522, out of all techniques. 


## References
[1] Ritesh Ahuja, Gabriel Ghinita, and Cyrus Shahabi. “Differentially-Private Next-Location Prediction with Neural Networks”. In: Proceedings of the 23rd International Conference on Extending Database
Technology, EDBT 2020, Copenhagen, Denmark, March 30 - April 02, 2020. Ed. by Angela Bonifati et
al. 

[2] Felipe T. Brito, André L. C. Mendonça, and Javam C. Machado. “A Differentially Private Guide for
Graph Analytics”. In: Proceedings 27th International Conference on Extending Database Technology,
EDBT 2024, Paestum, Italy, March 25 - March 28. Ed. by Letizia Tanca et al. OpenProceedings.org,
2024, pp. 850–853. doi: 10.48786/EDBT.2024.8610.48786/EDBT.2024.86. url: https://doi.org/10.48786/edbt.2024.86https://doi.org/10.48786/edbt.2024.86
(cit. on p. 11).

[3] Vishal Chakraborty et al. “Data-CASE: Grounding Data Regulations for Compliant Data Processing
Systems”. In: Proceedings 27th International Conference on Extending Database Technology, EDBT
2024, Paestum, Italy, March 25 - March 28. Ed. by Letizia Tanca et al.

[4] Vikram S. Chundawat et al. “Zero-Shot Machine Unlearning”. In: Trans. Info. For. Sec. 18 (Jan. 2023),pp. 2345–2354. issn: 1556-6013.

[5] Meghdad Kurmanji et al. Towards Unbounded Machine Unlearning. 2023. arXiv: 2302.09880 [cs.LG]2302.09880 [cs.LG].

[6] Sina Shaham, Gabriel Ghinita, and Cyrus Shahabi. “Differentially-Private Publication of Origin-Destination
Matrices with Intermediate Stops”. In: Proceedings of the 25th International Conference on Extending
Database Technology, EDBT 2022, Edinburgh, UK, March 29 - April 1, 2022. Ed. by Julia Stoyanovich et
al. 

[7] Ayush K. Tarun et al. “Fast Yet Effective Machine Unlearning”. In: IEEE Transactions on Neural Net-
works and Learning Systems 35.9 (Sept. 2024), pp. 13046–13055. issn: 2162-2388.

[8] Yuncheng Wu et al. “Privacy Preserving Group Nearest Neighbor Search”. In: Proceedings of the 21st International Conference on Extending Database Technology, EDBT 2018, Vienna, Austria, March 26-29, 2018. Ed. by Michael H. Böhlen et al.