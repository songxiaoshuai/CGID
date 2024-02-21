# Continual Generalized Intent Discovery: Marching Towards Dynamic and Open-world Intent Recognition</h2>

<p>
📃 <a href="https://aclanthology.org/2023.findings-emnlp.289.pdf">EMNLP2023 Paper</a>
  •
📃 <a href="https://arxiv.org/pdf/2310.10184.pdf">Arxiv Paper</a>
  •
🤗 <a href="https://huggingface.co/XXHStudyHard/PLRD-IND-Pretraining">HuggingFace Model</a> 
</p>


**Authors**:  Xiaoshuai Song,  Yutao Mou,  Keqing He, Yueyan Qiu, Pei Wang,  Weiran Xu

## Introduction
In a practical dialogue system, users may input out-of-domain (OOD) queries. The Generalized Intent Discovery (GID) task aims to discover OOD intents from OOD queries and extend them to the in-domain (IND) classifier. However, GID only considers one stage of OOD learning, and needs to utilize the data in all previous stages for joint training, which limits its wide application in reality. In this paper, we introduce a new task, Continual Generalized Intent Discovery (CGID), which aims to continuously and automatically discover OOD intents from dynamic OOD data streams and then incrementally add them to the classifier with almost no previous data, thus moving towards dynamic intent recognition in an open world. Next, we propose a method called Prototype-guided Learning with Replay and Distillation (PLRD) for CGID, which bootstraps new intent discovery through class prototypes and balances new and old intents through data replay and feature distillation. Finally, we conduct detailed experiments and analysis to verify the effectiveness of PLRD and understand the key challenges of CGID for future research.

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@misc{song2023continual,
      title={Continual Generalized Intent Discovery: Marching Towards Dynamic and Open-world Intent Recognition}, 
      author={Xiaoshuai Song and Yutao Mou and Keqing He and Yueyan Qiu and Pei Wang and Weiran Xu},
      year={2023},
      eprint={2310.10184},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
