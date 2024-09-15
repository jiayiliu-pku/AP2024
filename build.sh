#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

dockername='upload0827autopet'


docker build -t $dockername "$SCRIPTPATH"
