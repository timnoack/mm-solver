#!/bin/sh


cd "$(dirname "$0")"
rm -r cfd2 || true

echo "Downloading.."
wget https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/cfd2.tar.gz

echo "Extracting.."
tar -xvf cfd2.tar.gz

rm cfd2.tar.gz