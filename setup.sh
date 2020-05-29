pip3 install -r requirements.txt
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -O enron.tar.gz
tar xzvf enron.tar.gz
rm enron.tar.gz

mkdir apache_ml
cd apache_ml
for y in {2002..2011}; do
    for m in {01..12}; do
        wget "http://mail-archives.apache.org/mod_mbox/lucene-java-user/${y}${m}.mbox"
    done
done
