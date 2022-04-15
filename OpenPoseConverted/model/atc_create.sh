#!/bin/bash
echo input model is ${1}
echo input weights are ${2}
echo image size is ${3}, ${4}
echo output name is ./model/openposehand${3}${4}
atc --model=./${1} --weight=./${2} --framework=0 --output=./model/openposehand${3}${4} --soc_version=Ascend310 --input_shape="image:1,3,${3},${4}"
