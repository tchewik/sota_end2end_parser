apt install -y libffi-dev liblzma-dev
apt install -y software-properties-common
add-apt-repository ppa:openjdk-r/ppa
apt update && apt install -y openjdk-8-jdk
apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && apt update && apt install python3.7 python3.7-dev python3.7-venv

python3.7 -m venv venv
source venv/bin/activate
pip install -U pip wheel
pip install allennlp==1.0.0
pip install torch==1.3.1
pip install transformers==3.0.2
pip install stanfordcorenlp==3.9.1.1
pip install -U git+https://github.com/IINemo/isanlp.git
pip install spacy progressbar
python -m spacy download en_core_web_sm

mkdir data/elmo && cd data/elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
cd ../..

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
apt install -y zip
unzip stanford-corenlp-full-2018-02-27.zip
