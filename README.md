#### Reproducibility Project - SleepQA : A Health Coaching Dataset on sleep for Extractive Question Answering
- Written by Kihyuk Song and Hyunkyung Lee
- Subject: CS598 Deep Learning for Healthcare

##### 1. Citation to the original paper
Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Rei Ern Jaime Pang, Jessica Chen, Vaaruni Nayak, Shafiq Joty, Josip Car. SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering. Proceedings of Machine Learning for Health (ML4H) 2022 Workshop.

##### 2. Link to the original paper’s repo (if applicable)
https://github.com/IvaBojic/SleepQA

##### 3. Dependencies
(1) Installation by pip
"pyserini", "faiss-cpu>=1.6.1", "filelock", "numpy", "regex", "torch>=1.5.0", "transformers>=4.3", "tqdm>=4.27", "wget", "spacy>=2.1.8", "hydra-core>=1.0.0", "omegaconf>=2.0.1", "jsonlines", "soundfile", "editdistance"
(2) Installation by apt-get
openjdk-11-jdk, git-lfs

● Data download instruction
Datas are downloaded from https://github.com/IvaBojic/SleepQA
```
sleep_train:
  _target_: dpr.data.retriever_data.CsvQASrc
  file: "../../../../data/training/sleep-train.csv"

sleep_dev:
  _target_: dpr.data.retriever_data.CsvQASrc
  file: "../../../../data/training/sleep-dev.csv"

sleep_test:
  _target_: dpr.data.retriever_data.CsvQASrc
  file: "../../../../data/training/sleep-test.csv"

sleep_open:
  _target_: dpr.data.retriever_data.CsvQASrc
  file: "../../../../data/training/open_questions.csv"
```


● Preprocessing code + command 
I

● Training code + command (if applicable)
● Evaluation code + command (if applicable)
● Pretrained model (if applicable)
● Table of results (no need to include additional experiments, but main reproducibility
result should be included)
