
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
rm ~/miniconda.sh
eval "$(/root/miniconda/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false

conda install pytorch torchvision cpuonly -c pytorch -y
conda install faiss-cpu -c pytorch -y
conda install --file requirements.txt -y
cd GloVe && make && cd ..
