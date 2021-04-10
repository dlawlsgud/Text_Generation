# Text_Generation

> Text Summarization = Text를 입력받아 본문을 이해하고 중요한 정보만 정제해 문장을 생성하는 것이다.


>> ##### 1. 추출 요약 : 본문에서 중요하다고 생각되는 문장을 그대로 추출해 요약문을 생성한다.  
>> ##### 2. 추상 요약 : 본문을 이해하고 이를 잘 반영한 추상적인 요약 문장을 생성해 낸다. 
>> 
>> ##### Text summarization 기본 Architecture은 Encoder에 document를 넣고 Decoder에서 summary를 출력하는 "Sequence-to-Sequence"구조다.
>> ##### 최근 대량 코퍼스를 사용해 Self-Supervised Learning 시킨 Pretrained-model을 이용해 Downstream task에 맞게 Fine-tuning 시키는 Transfer Learning이 좋은 성능을 보여준다.
>> ##### 
