#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/text8
VECTOR_DATA=$DATA_DIR/text8-vector-test1.bin

if [ ! -e $VECTOR_DATA ]; then
  if [ ! -e $TEXT_DATA ]; then
		sh ./create-text8-data.sh
	fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -save-vocab ../data/text8-vocab.bin -cbow 1 -size 200 -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 8 -binary 1 -iter 15
  #time $BIN_DIR/word2vec
fi
