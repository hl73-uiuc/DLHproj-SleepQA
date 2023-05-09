## Reproducibility Project - SleepQA : A Health Coaching Dataset on sleep for Extractive Question Answering
- Written by Kihyuk Song and Hyunkyung Lee
- Subject: CS598 Deep Learning for Healthcare
- Jupyter notebook: DLHproj-SleepQA.ipynb

### 1. Citation to the original paper
Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Rei Ern Jaime Pang, Jessica Chen, Vaaruni Nayak, Shafiq Joty, Josip Car. SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering. Proceedings of Machine Learning for Health (ML4H) 2022 Workshop.

### 2. Link to the original paper’s repo
https://github.com/IvaBojic/SleepQA

### 3. Dependencies
(1) Installation by pip
"pyserini", "faiss-cpu>=1.6.1", "filelock", "numpy", "regex", "torch>=1.5.0", "transformers>=4.3", "tqdm>=4.27", "wget", "spacy>=2.1.8", "hydra-core>=1.0.0", "omegaconf>=2.0.1", "jsonlines", "soundfile", "editdistance"
(2) Installation by apt-get
openjdk-11-jdk, git-lfs

### 4. Data download instruction
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

### 5. Preprocessing code + command 
I built the docker image. 
```
### Dockerfile
FROM nvcr.io/nvidia/pytorch:21.11-py3
WORKDIR /data
COPY SleepQA/DPR-main /data/SleepQA/DPR-main
RUN pip install /data/SleepQA/DPR-main
RUN python -m spacy download en_core_web_sm
RUN pip install pyserini
RUN apt-get update && apt-get install -y openjdk-11-jdk
RUN apt-get install -y git-lfs
CMD ["/bin/bash"]
```
I created two containers using the image that can utilize 4 GPUs each.
```
docker build -t dpr-container .
docker run --gpus=4 --name dpr01 -v /DATA/tmpdata_hk:/data -itd dpr-container 
docker run --gpus '"device=4,5,6,7"' --name dpr02 -v /DATA/tmpdata_hk:/data -itd dpr-container 
```

### 6. Training code + command
DPR (Dense Passage Retrieval) is a framework that efficiently searches for information in large amounts of text. The Biencoder in DPR consists of a question encoder and a document encoder, which each convert the question and document into vectors to identify highly relevant documents. The Extractive Reader then extracts accurate answers to questions based on the documents returned by the Biencoder. Both the question and document are used as inputs to extract the answer.
I changed the hyperparameter from the yaml file of Biencoder and Extractive reader.
I used two models such as BioBERT and ClinicialBERT.
This stage is processed in `DPR-main` directory.
```
### biencoder conf
conf/biencoder_train_cfg.yaml
  - encoder: hf_biobert, hf_clinicalBERT

conf/train/biencoder_local.yaml
  - dev_batch_size: 32 (previous value was 16)
  - learning_rate: 2e-5
  - num_train_epochs: 20 (previous value was 30)
  
### extractive reader conf
conf/extractive_reader_train_cfg.yaml
  - encoder: hf_biobert, hf_clinicalBERT

conf/train/extractive_reader_default.yaml
  - dev_batch_size: 32 (previous value was 16)
  - learning_rate: 1e-5 (previous value was 2e-5)
  - num_train_epochs: 20 (previous value was 30)
```

I ran the training of the models.
I used the PyTorch to parallelize the computation across 4 GPUs.
```
### biencoder training
python -m torch.distributed.launch --nproc_per_node=4 \
train_dense_encoder.py \
train=biencoder_local \
train_datasets="/data/SleepQA/data/training/sleep-train.json" \
dev_datasets="/data/SleepQA/data/training/sleep-dev.json" \
output_dir="train_dense_encoder/"

### extractive reader training
python -m torch.distributed.launch --nproc_per_node=4 \
train_extractive_reader.py \
encoder.sequence_length=300 \
train_files="/data/SleepQA/data/training/oracle/sleep-train.json" \
dev_files="/data/SleepQA/data/training/oracle/sleep-dev.json"  \
output_dir="biobert/reader"
```

### 7. Evaluation code + command 
I saved the results of inferencing from trained models.
This stage is processed in `DPR-main` directory.
First, I extracted feature vectors with the biencoder.
```
python generate_dense_embeddings.py \
    model_file="/data/SleepQA/DPR-main/outputs/2023-05-08/14-14-01/train_dense_encoder/dpr_biencoder.19" \
    ctx_src="dpr_sleep" \
    out_file="/data/SleepQA/models/processed/encoder-clinical"   
```

Next, I searched for questions and save the results in a CSV file with the biencoder.
```
python dense_retriever.py \
    model_file="/data/SleepQA/DPR-main/outputs/2023-05-08/14-14-01/train_dense_encoder/dpr_biencoder.19"\
    encoded_ctx_files=["/data/SleepQA/models/processed/encoder-clinical_0"] \
    out_file="/data/SleepQA/models/processed/retriever-clinical.csv"
```

Finally, I saved the answers with the extractvie reader
```
python -m torch.distributed.launch --nproc_per_node=4 \
  train_extractive_reader.py \
  encoder.sequence_length=300 \
  passages_per_question_predict=100 \
  eval_top_docs=[10,20,40,50,80,100] \
  dev_files="/data/SleepQA/data/training/oracle/sleep-dev.json"\
  train.dev_batch_size=16 \
  model_file="/data/SleepQA/DPR-main/outputs/2023-05-08/09-02-18/ClinicalBERT/reader/dpr_extractive_reader.10.500" \
  prediction_results_file="/data/SleepQA/models/processed/reader-clinical.csv" 
```

This is the part of the results file. 
prediction_results_file="/data/SleepQA/models/processed/reader-clinical.csv"
```
    {
        "question": "how are flippable mattresses constructed?",
        "gold_answers": [
            "using a different comfort layer on each side of the support core, allowing either side of the mattress to be used as the top"
        ],
        "predictions": [
            {
                "top_k": 10,
                "prediction": {
                    "text": "using a different comfort layer on each side of the support core",
                    "score": 21.362292289733887,
                    "relevance_score": 3.7556169033050537,
                    "passage_idx": 0,
                    "passage": "flippable mattresses are constructed using a different comfort layer on each side of the support core, allowing either side of the mattress to be used as the top. most flippable mattresses have a different firmness level on each side. in flippable mattresses, the support core consists of the firmer layers in the middle of the mattress, as well as the comfort layers from the side that's placed face - down. the vast majority of mattresses have a support core containing either steel coils, high - density polyfoam, or latex. more rarely, shoppers may come across a model containing air or water chambers in the support core."
                }
            },
            {
                "top_k": 20,
                "prediction": {
                    "text": "using a different comfort layer on each side of the support core",
                    "score": 21.362292289733887,
                    "relevance_score": 3.7556169033050537,
                    "passage_idx": 0,
                    "passage": "flippable mattresses are constructed using a different comfort layer on each side of the support core, allowing either side of the mattress to be used as the top. most flippable mattresses have a different firmness level on each side. in flippable mattresses, the support core consists of the firmer layers in the middle of the mattress, as well as the comfort layers from the side that's placed face - down. the vast majority of mattresses have a support core containing either steel coils, high - density polyfoam, or latex. more rarely, shoppers may come across a model containing air or water chambers in the support core."
                }
            },
```


Then, I converted the DPR checkpoints to the PyTorch model.
This stage is processed in `models` directory.
```
python convert_dpr_original_checkpoint_to_pytorch.py --type question_encoder --src pipeline1/dpr_biencoder.19 --dest pytorch/question_encoder

python convert_dpr_original_checkpoint_to_pytorch.py --type reader --src pipeline1_baseline/cp_models/dpr_extractive_reader.7.59 --dest pytorch/reader
```

I made the QA pipeline system and evaluate this using the results csv file and PyTorch model.
This stage is processed in `models` and `eval` directory.
```
python qa_system.py
python eval/__main__.py
```

### 8. Pretrained model
The size of the pretrained models is too big (above 100GB) so that I could not upload it.

### 9. Table of results
This is the results of two models.
|Role|Model|Batch Size|Learning rate|Num train epochs|Avg runtime for each epoch|EM score|
|---|---|---|---|---|---|---|
|Biencoder|Clinical BERT|32|2e-5|20|4min 55sec||
|Extractive Reader|Clinical BERT|32|1e-5|20|2min 24sec|53.60|
|Biencoder|BioBERT|32|2e-5|20|4min 47sec||
|Extractive Reader|BioBERT|32|1e-5|20|2min 32sec|58.40|


|Role|Model|Batch Size|Learning rate|Num train epochs|Avg runtime for each epoch|EM score|
|---|---|---|---|---|---|---|
|Biencoder|BioBERT|32|2e-5|20|4min 47sec||
|Extractive Reader|BioBERT|32|1e-5|20|2min 32sec|58.40|
