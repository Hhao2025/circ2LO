import math
import pandas as pd
import torch
import os
from torch import nn
import collections
from d2l import torch as d2l
import numpy as np
from einops import rearrange 

def tokenize_nmt(text, num_examples=None):
    """词元化数据数据集。"""
    source = []
    for i, line in enumerate(text):
        if num_examples and i > num_examples:
            break
        a=list(line)
        source.append(a[:-1])
    return source

def reverSeq(seq):
    seq = list(seq)
    seq.reverse()
    basePair = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N'}
    seq = [basePair[i] for i in seq]
    seq = "".join(seq)
    return seq

class PositionalEncoding(nn.Module):#位置编码
    def __init__(self, num_hiddens, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,torch.arange(0, num_hiddens, 2, dtype=torch.float32) /num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P=rearrange(self.P,'b l f -> b f l')
    def forward(self, X):
        #print(X.shape)
        '''very amazing method'''
        #self.P[:,int(X.shape[1]/2):,:]=-self.P[:,int(X.shape[1]/2):,:]
        #X = X + self.P[:, :X.shape[1], 0].T.to(X.device)

        X = X + self.P[:, 1:2, :].to(X.device)
        print(self.P[:,:3,:])
        print(self.P.shape)
        return X

class PositionalEncod(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncod, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

def extractknt(bedFile,gene,fastafile):
    import pysam
    seqFile = bedFile.replace(".bed", "seq.txt")
    data=[]
   # fasta1 = "data/data.fasta" dfffffffffffffffffffdfdfdddffd
    #genome = pysam.FastaFile("data/GRCh37.p13.genome.fa")
    # hg38.fa acc:0.795
    # GRCh37.p13.genome.fa   acc:0.944
    with open(fastafile, 'w') as fasta, open(bedFile, 'r') as bed, open(seqFile, 'w') as seq:
        genome = pysam.FastaFile(gene)
        i=0
        for line in bed:
            values = line.split()
            if not values[1].isdigit():
                print(values[1]+"not a ditgit")
                continue
            chr = values[0]
            start = int(values[1]) - 50 
            end = int(values[2]) + 50
            strand = values[3]
            #seqData = genome.fetch(chr, start, end).upper()
            try:
                seqData = genome.fetch(chr, start, end).upper()
            except:
                print(chr+"not in")
                continue
            seqData =seqData[:100] + seqData[-100:] #20,30,40,50,60,70
            if strand == '-':
                seqData = reverSeq(seqData)
            if len(seqData) == 0 or seqData[0] == 'N' or seqData[-1] == 'N':
                continue
            #seq.write('>char'+str(i)+'\n')
            i+=1
            seq.write(seqData + '\n')


            #fasta1.write('>' + ':'.join([chr, str(start), str(end), strand]) + '\n')
            fasta.write('>' + ':'.join([chr, str(start), str(end), strand]) + '\n')
            fasta.write(seqData + '\n')
            data.append(line)
    print("Extract 200nt successfully!\n")
    return seqFile,data

def TranFea(file1,file2):
    f=open(file1,'r')
    f1=open(file2,'r')
    pos=f.readlines()
    neg=f1.readlines()
    f.close()
    f1.close()
    spos=tokenize_nmt(pos)
    sneg=tokenize_nmt(neg)
    vocab=d2l.Vocab(spos)
    vpos=[vocab[i] for i in spos]
    vneg=[vocab[i] for i in sneg]
    vpos=torch.Tensor(vpos)
    vneg=torch.Tensor(vneg)
    #k=vpos.reshape(-1,200,1)
    k=vpos.reshape(-1,1,200)
    #k2=vneg.reshape(-1,200,1)
    k2=vneg.reshape(-1,1,200)
    #pos_encoding = PositionalEncoding(200, 0.01)
    #pos_encoding = PositionalEncod(200, 0.01)
    print(k.shape)
    #pos_encoding.eval()
   #位置编码
    #X = pos_encoding(k)
    #X2 = pos_encoding(k2)
    # 无位置编码
    X = k
    X2 = k2
    #X=X.reshape(-1,1,200).squeeze()
    X=X.squeeze()
    #X2=X2.reshape(-1,1,200).squeeze()
    X2=X2.squeeze()
    a=X.numpy()
    #print(a[0])
    #print(a.shape)
    b=X2.numpy()
    feaFile1 = file1[:file1.find(".")] + ".fea"
    feaFile2 = file2[:file2.find(".")] + ".fea"
    with open(feaFile1,'w') as f1,open(feaFile2,'w') as f2:
        for line in a:
            f1.write(" ".join('%f' %id for id in line)+'\n')
        for line in b:
            f2.write(" ".join('%f' %id for id in line)+'\n')
    print(f1.name)
    print(f2.name)
    print('get TranFea successful')
    return feaFile1,feaFile2
