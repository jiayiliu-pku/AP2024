#!/bin/bash

./build.sh

dockername='upload0827autopet'
docker save $dockername | gzip -c > ${dockername}.tar.gz
