apt install -y libffi-dev liblzma-dev
apt install -y software-properties-common
add-apt-repository ppa:openjdk-r/ppa
apt update && apt install -y openjdk-8-jdk
apt install software-properties-common -y

curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
bash
pyenv install 3.6.9
pyenv virtualenv 3.6.9 py369-rst
pyenv activate py369-rst
pip install ipykernel && python -m ipykernel install --name "py369-rst"

pip install -U pip wheel
pip install allennlp==1.0.0
pip install torch==1.3.1
pip install transformers==3.0.2
pip install stanfordcorenlp==3.9.1.1
pip install -U git+https://github.com/IINemo/isanlp.git

# For tokenization (option 1, spacy)
# pip install -U spacy
# python -m spacy download en_core_web_sm
#pip install -U numpy==1.20.3

# For tokenization (option 2, udpipe)
pip install ufal.udpipe
cd data && wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe
pip install -r all_requirements.txt

mkdir data/elmo && cd data/elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
cd ../..

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
apt install -y zip
unzip stanford-corenlp-full-2018-02-27.zip
