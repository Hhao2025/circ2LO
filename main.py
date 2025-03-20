import argparse
import os
import dataProcessing
import extractFeatures
import runModel
import TransformerFea
import Stream2Model
import Stream2Modelgcforest
"""
one example


"""
parser = argparse.ArgumentParser("Input of the model")
parser.add_argument("--train", type=bool, default=True, help="The option of training model")
parser.add_argument("--predict", type=bool, default=False, help="The option of predicting circRNA")
parser.add_argument("--data_dir",type=str,default="data/",help="Input the directory of data")
#拟南介
# parser.add_argument("--pos_data", type=str, default="data/nnj_circRNA_new.bed", help="The filename of positive data")
# parser.add_argument("--neg_data", type=str, default="data/nnj_lncRNA_new.bed", help="The filename of negative data")
# parser.add_argument("--genome", type=str, default="data/TAIR10_chr_all.fas", help="The fasta file of genome")
#人類
parser.add_argument("--pos_data", type=str, default="data/human_circRNA.bed", help="The filename of positive data")
parser.add_argument("--neg_data", type=str, default="data/human_lncRNAclean.bed", help="The filename of negative data")
# parser.add_argument("--pos_data", type=str, default="data/human_circRNA_new092.bed", help="The filename of positive data")
# parser.add_argument("--neg_data", type=str, default="data/human_lncRNA_new.bed", help="The filename of negative data")

parser.add_argument("--genome", type=str, default="data/hg38.fasta", help="The fasta file of genome")
# parser.add_argument("--genome", type=str, default="data/GRCh37.p13.genome.fa", help="The fasta file of genome")

#parser.add_argument("--genome", type=str, default="data/hg38.phastCons20way.bw", help="The fasta file of genome")

#小鼠
#parser.add_argument("--pos_data", type=str, default="data/mus_circRNA.bed", help="The filename of positive data")
#parser.add_argument("--neg_data", type=str, default="data/mus_lncRNA.bed", help="The filename of negative data")
# parser.add_argument("--pos_data", type=str, default="data/mouse_posRNA.bed", help="The filename of positive data")
# parser.add_argument("--neg_data", type=str, default="data/mouse_negRNA.bed", help="The filename of negative data")
# parser.add_argument("--genome", type=str, default="data/mouse.fa", help="The fasta file of genome")

#hg38.fa acc:0.79743 > dcc
#you should change genome file in Transformfea.py
parser.add_argument("--model", type=str, default="df", help="The name of model")
parser.add_argument("--result", type=str, default="result.txt", help="The result of prediction")
# if not os.path.exists(parser.data_dir):
#     os.mkdir(parser.data_dir)     data路径必然有，这里放个新建文件夹的例子
    
print('This is a program designed for detecting circRNAs in tha using deepforest.\n')
input = parser.parse_args()
dataProcessing.rmBedDup(input.data_dir, input.pos_data, input.neg_data) # remove duplication and find no dup
posf,posbed=TransformerFea.extractknt(input.pos_data,input.genome,"data/human_pos_data.fasta")
negf,negbed=TransformerFea.extractknt(input.neg_data,input.genome,"data/human_neg_data.fasta")

