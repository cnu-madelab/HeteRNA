
### Dataset:

- KGRACDA dataset2 train.txt , test.txt , valid.txt

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
