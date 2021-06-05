#!/bin/bash
rm -rf release
mkdir -p release

cp -rf Analysis *.{hpp,cpp,txt} release/

VERSION=$(git tag)
cat addon.json | sed  "s/\$VERSION/$VERSION/g" > release/addon.json

mkdir -p release/3rdparty/Gist
cp -rf 3rdparty/Gist/src \
       3rdparty/Gist/LICENSE.txt \
       release/3rdparty/Gist/

mv release score-addon-analysis
7z a score-addon-analysis.zip score-addon-analysis
