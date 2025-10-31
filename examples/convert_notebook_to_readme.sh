#/bin/bash
python -m nbconvert --to markdown CIE.ipynb
cp ../README_base.md ../README.md
cat CIE.md >> ../README.md
mkdir ../CIE_files
mv CIE_files/* ../CIE_files


python -m nbconvert --to rst CIE.ipynb
cp CIE.rst ../docs/source/Quickstart.rst
cp -r ../CIE_files/* ../docs/source/CIE_files
