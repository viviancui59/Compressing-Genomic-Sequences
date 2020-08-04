# -*- encoding: utf-8 -*-
import os
import os.path
import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
# filename ='SRR642636_1.fq'# downloaded multi-fasta file from MITOMAP database

reads = []
# , "rU"

for i in range(2672): #2672 is the number of fasta
    filename = './fish_xlt/' + str(i) + '.fasta'  # downloaded multi-fasta file from MITOMAP database
    with open(filename, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta") :
            

                reads.append(record)


print(len(reads))
train_set_size=int(len(reads)*0.7)
valid_set_size=int(len(reads)*0.2)
test_set_size=int(len(reads)*0.1)

def split_train_test(data, test_ratio,valid_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    valid_set_size=int(test_set_size*valid_ratio/(valid_ratio+test_ratio))
    test_indices = shuffled_indices[valid_set_size:test_set_size]
    valid_indices = shuffled_indices[:valid_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[valid_indices], data[test_indices]
items = [x for x in range(len(reads))]

random.shuffle(items)

train = items[0:train_set_size]
valid = items[train_set_size:train_set_size+valid_set_size]
test = items[train_set_size+valid_set_size:]
train_record = []
valid_record = []
test_record = []
# train_set, valid_set,test_set = split_train_test(reads,0.2,0.1)

def check(read):
    record = read
    my_dna = str(read.seq.upper())
    for i, base in enumerate(my_dna):
        if base not in 'ACGTN':
            my_dna = my_dna.replace(base,'N')
    record.seq= Seq(my_dna, generic_dna)
    for i, base in enumerate(record.seq):
        if base not in 'ACGTN':
            print(record.seq[i])
    return record
for i in train:
    read = check(reads[i])
    train_record.append(read)
for i in valid:
    read = check(reads[i])
    valid_record.append(read)
for i in test:
    read = check(reads[i])
    test_record.append(read)
print(len(train_record), "train +", len(test_record), "test")
#save the data
SeqIO.write(train_record, "./fna_data/train.fasta", "fasta")
SeqIO.write(valid_record, "./fna_data/valid.fasta", "fasta")
SeqIO.write(test_record, "./fna_data/test.fasta", "fasta")


def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    text = ""
    for i,record in enumerate(records):
       # text += str(record.seq)
        print("No."+str(i)+": "+record.seq)
    #return text


#read_fasta("fq_valid.fasta")

