
wget https://datahub.io/machine-learning/phoneme/r/phoneme.csv
echo "$(tail -n +2 phoneme.csv)" > phoneme.csv

mkdir landsat
cd landsat
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst
cd ..

mkdir spambase
cd spambase
wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
cd ..

mkdir credit
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00573/SouthGermanCredit.zip -P credit
cd credit
unzip SouthGermanCredit.zip
cd ..
echo "Ensure that SouthGermanCredit.zip has unzipped successfully in the credit directory before proceeding"
