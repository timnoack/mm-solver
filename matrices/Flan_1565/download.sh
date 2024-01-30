#!/bin/sh


cd "$(dirname "$0")"
rm -r Flan_1565 || true

echo "Downloading.."
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Flan_1565.tar.gz

echo "Extracting.."
tar -xvf Flan_1565.tar.gz

rm Flan_1565.tar.gz