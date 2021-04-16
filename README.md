# Text_Generation

### Text Summarization 
#### : 길이가 긴 본문의 의미는 유지한 채 본문보다 짧은 문장을 생성하는 자연어 처리 태스크다. 
> ##### 1. 추출 요약 : 본문에서 중요하다고 생각되는 단어, 구, 또는 문장을 그대로 추출해 문장을 생성한다.  
> ##### 2. 추상 요약 : 본문 전체 내용을 이해하고 이해한 내용을 바탕으로 새로운 문장을 생성한다.
##### - Text summarization 기본 Architecture은 Encoder에 document를 넣고 Decoder에서 summary를 출력하는 "Sequence-to-Sequence"구조다.
##### - 최근 대량 코퍼스를 사용해 Self-Supervised Learning 시킨 Pretrained-model을 이용해 Downstream task에 맞게 Fine-tuning 시키는 Transfer Learning이 좋은 성능을 보여준다.

### 연구주제 => Curriculum Learning을 적용한 생성요약


### Itroduction

생성요약은 추출요약보다 허위정보 생성, 잘못된 개체명 인식 등의 본문 전체 내용에 대한 이해 부족으로 비롯된 신뢰성 부분에 단점이 있다. 문법적으로 정확하고 가독성이 좋은 새로운 문장을 생성하는 능력도 중요하지만, 본문 전체 내용에 대한 이해를 바탕으로 한 중요 문장 인식 능력 또한 필요하다. 본 논문에서는 **추출요약 학습을 통해 중요 문장을 인식하는 능력을 갖춘 모델에 생성요약을 학습**시켜 기존 생성요약의 단점을 보완하고자 한다.<br>

