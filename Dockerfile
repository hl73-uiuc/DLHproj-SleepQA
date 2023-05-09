FROM nvcr.io/nvidia/pytorch:21.11-py3
WORKDIR /data
COPY SleepQA/DPR-main /data/SleepQA/DPR-main
RUN pip install pyserini
RUN pip install /data/SleepQA/DPR-main
RUN python -m spacy download en_core_web_sm
RUN apt-get update && apt-get install -y openjdk-11-jdk
RUN apt-get install -y git-lfs
CMD ["/bin/bash"]
