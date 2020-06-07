FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime


RUN apt-get update && \
	apt-get install -qqy git libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install deeppavlov git+https://github.com/IINemo/isanlp.git conllu git+https://github.com/chorusai/brave.git \
    easydict toposort networkx
RUN python -m deeppavlov install syntax_ru_syntagrus_bert morpho_ru_syntagrus_pymorphy


ENV PYTHONPATH=/workspace
VOLUME /workspace
VOLUME /data
