# Korean-Semantic-Classifier
using ai-hub datasets for training

## Details
* Korean_semantic_classification_Dataloader.\* : torchtext를 사용해 train, valid loader를 만든다.
* Korean_semantic_classification_utils.\* : 기울기(grad), 파라미터(parameter)에 norm을 계산한다.(default norm = 2)
* Korean_semantic_classification_RNN_model.\* : gru layer를 사용해 모델을 구성.
* Korean_semantic_classification_train.\* : Dataloader, RNN_model, train_loader를 사용해 모델을 학습시킨다.
* Korean_semantic_classification_trainer.\* : ignite engine을 사용한다.
