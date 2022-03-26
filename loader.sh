#!/bin/bash

# load seq2vec model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ibyDaM9OkLCW1GcYnDaoKmoOTwcluKeT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ibyDaM9OkLCW1GcYnDaoKmoOTwcluKeT" -O tmpfile && rm -rf /tmp/cookies.txt
unzip -o tmpfile -d tmptmptmp
mkdir seq2vec/trained
mv tmptmptmp/checkpoints/yelp/daae/* seq2vec/trained/.
rm -r tmpfile
rm -r tmptmptmp

# load bookcorpus to evaluate world model
f='16KCjV9z_FHm8LgZw05RSuk4EsAWPOP_z'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${f} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${f}" -O tmpfile && rm -rf /tmp/cookies.txt
tar -jxvf tmpfile books_large_p2.txt
mkdir dataset
mv books_large_* dataset/.
rm tmpfile