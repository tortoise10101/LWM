#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ibyDaM9OkLCW1GcYnDaoKmoOTwcluKeT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ibyDaM9OkLCW1GcYnDaoKmoOTwcluKeT" -O tmpfile && rm -rf /tmp/cookies.txt
unzip -o tmpfile -d tmptmptmp
mkdir seq2vec/trained
mv tmptmptmp/checkpoints/yelp/daae/* seq2vec/trained/.
rm -r tmpfile
rm -r tmptmptmp