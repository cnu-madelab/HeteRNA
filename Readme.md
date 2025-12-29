### Overview
Heterogeneous ceRNA Network를 활용한 GNN 기반 circRNA–질병 연관성(Link Prediction) 예측
circRNA, miRNA, lncRNA, disease, gene 간의 이질적 생물학적 상호작용을 그래프로 모델링
Negative set 없이 link prediction을 수행하는 코드임

모델 학습 및 평가는 Runner 클래스를 중심으로 구성되어 있으며, 데이터 로딩, 그래프 구성, 모델 초기화, 학습, 검증, 테스트, 성능 평가가 하나의 일관된 파이프라인으로 관리됨.
Link prediction 성능은 MRR(Mean Reciprocal Rank)을 기준으로 평가되며, 동시에 circRNA–disease 예측 결과를 이진 분류 관점에서 해석하기 위해 Precision, Recall, F1-score, ROC-AUC, AUPR 등의 지표도 함께 계산됨

학습 과정에서는 (source node, relation)을 입력으로 받아, 해당 source node가 연결될 가능성이 있는 모든 object node에 대한 확률 분포를 예측함. 
이를 통해 circRNA–disease 관계를 link prediction 문제로 다루며, 생물학적으로 정의하기 어려운 negative set을 생성하지 않고도 예측이 가능하도록 설계됨

### Key feature
-Heterogeneous ceRNA Network 기반 GNN circRNA–질병 link prediction
-circRNA·miRNA·lncRNA·disease·gene 간 이질적 상호작용 그래프 모델링
-Negative set 없이 link prediction 방식으로 연관성 예측
-Runner 클래스 중심의 데이터 로딩–학습–검증–테스트–평가 파이프라인



### Dataset:

- KGRACDA dataset2 train.txt , test.txt , valid.txt
- 기존 KGRACDA 데이터셋을 기반으로 확장된 ceRNA network 구축 
- 핵심 lncRNA로 MALAT1을 중심으로 gene 관계 추가 

총 relation 수: 25,617
circRNA–disease: 1,339
miRNA–disease: 10,154
lncRNA–disease: 3,280
circRNA–miRNA: 1,226
miRNA–lncRNA: 9,506
lncRNA–gene: 94
lncRNA–ceRNA: 18 

Train / Valid / Test split:
Train: 25,188
Valid: 1,398
Test: 1,398

### Training model:
```shell
python NOT_Edit_CompGCN_run.py -score_func conve -opn corr

2024-10-12 08:36:19,086 - [INFO] - Early Stopping!!
2024-10-12 08:36:19,086 - [INFO] - Loading best model, Evaluating on Test data
2024-10-12 08:36:19,265 - [INFO] - [Test, Tail_Batch Step 0]    testrun_12_10_2024_08:34:53
2024-10-12 08:36:19,615 - [INFO] - [Test, Head_Batch Step 0]    testrun_12_10_2024_08:34:53
2024-10-12 08:36:19,742 - [INFO] - [Epoch 73 test]: MRR: Tail : 0.15893, Head : 0.09737, Avg : 0.12815

  - `-score_func` denotes the link prediction score score function
    -conve : 
    -distmult:
    -transe : 
  - `-opn` is the composition operation used in **CompGCN**. It can take the following values:
    - `sub` for subtraction operation:  Φ(e_s, e_r) = e_s - e_r
    - `mult` for multiplication operation:  Φ(e_s, e_r) = e_s * e_r
    - `corr` for circular-correlation: Φ(e_s, e_r) = e_s ★ e_r
  - `-name` is some name given for the run (used for storing model parameters)
  - `-model` is name of the model `compgcn'.
  - `-gpu` for specifying the GPU to use
  - Rest of the arguments can be listed using `python NOT_Edit_CompGCN_run.py -h`
  - 
### Citation:
Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{
    vashishth2020compositionbased,
    title={Composition-based Multi-Relational Graph Convolutional Networks},
    author={Shikhar Vashishth and Soumya Sanyal and Vikram Nitin and Partha Talukdar},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BylA_C4tPr}
}
```
For any clarification, comments, or suggestions please create an issue or contact [Shikhar](http://shikhar-vashishth.github.io).
