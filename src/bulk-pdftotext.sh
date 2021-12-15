#!/bin/bash

filepaths=`ls $1/*.pdf`
echo "Extracting text of all PDF files in $1"
for x in $filepaths; do pdftotext $x; done