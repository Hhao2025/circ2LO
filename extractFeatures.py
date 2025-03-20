# Extract features

import pyBigWig   # consFeatures saved by bigwig, which is read by pyBigWig
import os
import pickle
import numpy as np
from collections import defaultdict
from functools import reduce


def kmerFeatures(seqFile, k = 3):
    # Extract k-mer feature
    print("--------------------------------------------------------------------------------")
    print("Extract {}-mer features from ".format(k) + seqFile)
    kmerFile = seqFile[:seqFile.find(".")] + '.' + str(k) + 'mer'
    if os.path.exists(kmerFile):
        print(kmerFile + " has been created\n")
        #return kmerFile
    
    keys = reduce(lambda x, y: [i + j for i in x for j in y], [["A", "T", "C", "G"]] * k)
    with open(seqFile, 'r') as seqs, open(kmerFile, 'w') as fkmer:
        for seq in seqs.readlines():
            dic = defaultdict(int)
            length = len(seq)
            for i in range(0, length - k + 1):
                sub = seq[i: i + k]
                dic[sub] += 1
            for i in keys[:-1]:
                # Deal till the second from the bottom, the last special deal, + \n
                fkmer.write(str(dic[i] / length) + " ")
            fkmer.write(str(dic[keys[-1]] / length) + "\n")
            
    print("\nExtract k-mer features successfully!\n")
    return kmerFile

def onehotkmerFea(seqFile, k = 3):
    # Extract k-mer feature
    print("--------------------------------------------------------------------------------")
    print("Extract {}-mer features from ".format(k) + seqFile)
    kmerFile = seqFile[:seqFile.find(".")] + '.' + str(k) + 'ohmer'
    if os.path.exists(kmerFile):
        print(kmerFile + " has been created\n")
        #return kmerFile

    keys = reduce(lambda x, y: [i + j for i in x for j in y], [["A", "T", "C", "G"]] * k)
    with open(seqFile, 'r') as seqs, open(kmerFile, 'w') as fkmer:
        for seq in seqs.readlines():
            dic = defaultdict(int)
            length = len(seq)
            for i in range(0, length - k + 1):
                sub = seq[i: i + k]
                dic[sub] = 1
            for i in keys[:-1]:
                # Deal till the second from the bottom, the last special deal, + \n
                fkmer.write(str(dic[i]) + " ")
            fkmer.write(str(dic[keys[-1]]) + "\n")

    print("\nExtract onehot k-mer features successfully!\n")
    return kmerFile

def componentFeature(seqFile):
    print("--------------------------------------------------------------------------------")
    print("Extract component features from " + seqFile)
    compFile = seqFile[:seqFile.find(".")] + '.comp'
    if os.path.exists(compFile):
        print(compFile + " has been created\n")
        #return compFile
    
    
    with open(seqFile, 'r') as f, open(compFile, 'w') as comp:
        for line in f.readlines():
            # count AG GT GTAG AGGT percentage
            length = len(line)
            AG = line.count('AG') / length
            GT = line.count('GT') / length
            GTAG = line.count('GTAG') / length
            AGGT = line.count('AGGT') / length
            # count G + C
            G = line.count('G')
            C = line.count('C')
            GC = (G + C) / length
            comp.write(str(AG) + ' ' + str(GT) + ' ' + str(GTAG) + ' ' + str(AGGT) + ' ' + str(GC) + '\n')
    
    print("\nExtract component features successfully!\n")
    return compFile

def repeatFeature(fastaFile):
    print("--------------------------------------------------------------------------------")
    print("Extract tandem repeats features from " + fastaFile)
    repFile = fastaFile[:fastaFile.find(".")] + '.rep'
    if os.path.exists(repFile):
        print(repFile + " has been created\n")
        #return repFile
    
    with open(fastaFile, 'r') as fas, open(repFile, 'w') as frep:
        lines = fas.readlines()
        for i in range(0, len(lines), 2):
            fastaSub = fastaFile + ".sub"
            with open(fastaSub, 'w') as sub:
                sub.write(lines[i])
                sub.write(lines[i + 1])
        
            cmdline = "tool/trf409.linux64 " + fastaSub + " 2 3 5 80 10 20 2000 -f -d - m"
            runcmd = os.popen(cmdline, 'r')
            runcmd.close()
    
            htmls = [i for i in os.listdir(".") if i.endswith(".html")]  # trf409 will create several temp html files
            for i in htmls:
                os.remove(i)

            trfout = fastaSub.lstrip('data/') + '.2.3.5.80.10.20.2000.dat'  # .dat file has the caculated result through trf409
            with open(trfout, 'r') as f:
                cnt = [0] * 12
                flag = False
                for line in f.readlines():
                    if line.find('Sequence:') != -1:
                        continue
                    elif line.find('Parameters:') != -1:
                        flag = True
                        continue
                    elif flag and len(line) > 2:
                        values = line.split()
                        trfscores = values[2:13]
                        cnt[0] = cnt[0] + 1
                        for i in range(11):
                            cnt[i + 1] = cnt[i + 1] + float(trfscores[i])
            os.remove(trfout)
            frep.write('{:.3f}\t'.format(cnt[0]))
            if cnt[0] > 0:
                for i in cnt[1:]:
                    frep.write('{:.3f}\t'.format(float(i) / cnt[0]))
            else:
                for i in cnt[1:]:
                    frep.write('{:.3f}\t'.format(float(i)))
            frep.write('\n')
        os.remove(fastaSub)
    print("\nExtract tandem repeats features successfully!\n")
    return repFile

def codingFeature(fastaFile):
    print("--------------------------------------------------------------------------------")
    print("Extract coding potential features from " + fastaFile)
    codingFile = fastaFile[:fastaFile.find(".")] + '.coding'
    if os.path.exists(codingFile):
        print(codingFile + " has been created\n")
        #return codingFile
    
    cdftmp = codingFile + ".tmp"
    cmdline = 'tool/txCdsPredict ' + fastaFile + ' ' + cdftmp  # Caculate coding ability through ORF
    runcmd = os.popen(cmdline, 'r')
    runcmd.close()
    with open(codingFile, 'w') as cf, open(cdftmp, 'r') as f, open(fastaFile, 'r') as fas:
        faslines = fas.readlines()
        i = 0
        ftmp = open(fastaFile + '.not', 'w')
        for line in f.readlines():
            values = line.split()
            length = values[0].split(':')
            length = int(length[2]) - int(length[1])
            while faslines[i].strip('>').strip() != values[0]:
                ftmp.write(faslines[i].strip('>'))
                tmp = faslines[i].split(':')
                cf.write('0 0 ' + str(length) + ' ' + str(0) + '\n')
                i += 2
            i += 2
            values[10] = values[10].split(',')[0]
            # ORF score, ORF length, length, percent
            cf.write(values[5] + ' ' + values[10] + ' ' + str(length) + ' ' + str(float(values[10]) / length) + '\n')
    os.remove(cdftmp)
    print("\nExtract coding potential features successfully!\n")
    return codingFile

    
def conservationFeatures(bedFile, bigwigFile, gtfFile, modelDir):
    print("--------------------------------------------------------------------------------")
    consFile = bedFile + '.cons.txt'
    if os.path.exists(consFile):
        print(consFile + " has been created\n")
        #return consFile
    
    
    print("Extract conservation features from " + bedFile)
    print("Read gtf file to generate all exons")
    if os.path.exists(modelDir + "exons.pkl"):
        with open(modelDir + "exons.pkl", 'rb') as f:
            exons = pickle.load(f)
    else:
        with open(gtfFile, 'r') as gtf:
            # gtf 文件格式
            # #!genome-build GRCh38.p13
            # #!genome-version GRCh38
            # #!genome-date 2013-12
            # #!genome-build-accession NCBI:GCA_000001405.28
            # #!genebuild-last-updated 2020-08
            # 1	havana	exon	11869	12227	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; transcript_id "ENST00000456328"; transcript_version "2"; exon_number "1"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; exon_id "ENSE00002234944"; exon_version "1"; tag "basic"; transcript_support_level "1";
            exons = defaultdict(list)
            for line in gtf:
                if line[0] != '#':
                    gtfData = line.strip().split()
                    if len(gtfData) > 2 and gtfData[2] == 'exon':
                        chr = "chr" + gtfData[0]
                        start = int(gtfData[3])
                        end = int(gtfData[4])
                        strand = (gtfData[6])
                        exons[chr + strand].append([start, end])
        with open(modelDir + "exons.pkl", 'wb') as f:
            pickle.dump(exons, f)
    print("OK!\n")
    import pysam
    with open(bedFile, 'r') as bed,open(bedFile+"clean",'w') as f:
        genome = pysam.FastaFile("data/GRCh37.p13.genome.fa")
        for line in bed:
            bedData = line.split()
            if len(bedData) != 4:
                print("this bedData out of range:")
                print(line)
                continue
            if not bedData[1].isdigit():#筛选出不合法的数据
                print(bedData[1]+"not a ditgit")
                continue
            chr = bedData[0]
            start = int(bedData[1])
            end = int(bedData[2])
            strand = bedData[3]
            try:#筛选出不在范围内的
                seqData = genome.fetch(chr, start, end).upper()
                f.write(line+"\n")
                print(1)
            except:
                print(chr+"not in")
                continue
            if seqData.count('N') == len(seqData):
                print("This sequence is all 'N': " + line)
                continue
            exonArea = []
            flag = True
            for i in exons[chr + strand]:
                se = i[0]
                ee = i[1]
                if min(end, ee) - max(start, se) > 0:   # 两者有重叠
                    for j in range(len(exonArea)):
                        if exonArea[j][0] == se and exonArea[j][1] == ee:
                            flag = False
                            break
                    if flag:
                        exonArea.append((se, ee))
            sorted(exonArea)
            
            # 将重叠部分合并
            tmp = []
            for l, r in exonArea:
                if tmp and tmp[-1][1] >= l - 1:
                    tmp[-1][1] = max(tmp[-1][1], r)
                else:
                    tmp.append([l, r])
            exonArea = tmp
            if exonArea == []:
                exonArea.append([start, end])
            
            # 计算平均测序深度
            scores = []
            bigwig = pyBigWig.open(bigwigFile)
            for i in exonArea:
                try:
                    areas = bigwig.intervals(chr, i[0], i[1])
                    sum = 0
                    for j in areas:
                        sum += j[2]
                    scores.append(sum / (end - start))
                except:
                    with open(bedFile + "_None", 'a') as no:
                        no.write(chr + " " + str(i[0]) + " " + str(i[1]) + "\n")
                    scores.append(0)
            
            scores = np.array(scores)
            scoresMax = scores.max()
            scoresMin = scores.min()
            scoresMean = scores.mean()
            with open(consFile, 'a+') as cons:
                cons.write(str(scoresMax) + ' ' + str(scoresMin) + ' ' + str(scoresMean))

            scores = []
            try:
                areas = bigwig.intervals(chr, start, end)
                for i in areas:
                    scores.append(i[2])
            except:
                for i in range(end - start):
                    scores.append(0)
                    
            def getContinuousConservation(scores, k):
                # 连续保守度得分
                length = len(scores)
                if length==0:
                    return 0,0,0,0,0
                scores = np.array(scores)
                booleans = scores > k
                tmp = ""
                for i in booleans:
                    if i:
                        tmp = tmp + '1'
                    else:
                        tmp = tmp + '0'
                c5 = tmp.count("11111") / length
                c6 = tmp.count("111111") / length
                c7 = tmp.count("1111111") / length
                c8 = tmp.count("11111111") / length
                c9 = tmp.count("111111111") / length
                return c5, c6, c7, c8, c9
            
            c5, c6, c7, c8, c9 = getContinuousConservation(scores, 0.5)
            with open(consFile, 'a+') as cons:
                cons.write(' ' + str(c5) + ' ' + str(c6) + ' ' + str(c7) + ' ' + str(c8) + ' ' + str(c9))
            c5, c6, c7, c8, c9 = getContinuousConservation(scores, 0.6)
            with open(consFile, 'a+') as cons:
                cons.write(' ' + str(c5) + ' ' + str(c6) + ' ' + str(c7) + ' ' + str(c8) + ' ' + str(c9))
            c5, c6, c7, c8, c9 = getContinuousConservation(scores, 0.7)
            with open(consFile, 'a+') as cons:
                cons.write(' ' + str(c5) + ' ' + str(c6) + ' ' + str(c7) + ' ' + str(c8) + ' ' + str(c9))
            c5, c6, c7, c8, c9 = getContinuousConservation(scores, 0.8)
            with open(consFile, 'a+') as cons:
                cons.write(' ' + str(c5) + ' ' + str(c6) + ' ' + str(c7) + ' ' + str(c8) + ' ' + str(c9))
            c5, c6, c7, c8, c9 = getContinuousConservation(scores, 0.9)
            with open(consFile, 'a+') as cons:
                cons.write(' ' + str(c5) + ' ' + str(c6) + ' ' + str(c7) + ' ' + str(c8) + ' ' + str(c9))
                cons.write('\n')
    print("\nExtract conservation features successfully!\n")
    return consFile

def reverseComplementMatchingFeatures(fastaFile, genomeFile):
    print("--------------------------------------------------------------------------------")
    print("Extract reverse complement matching features from " + fastaFile)
    
    rcmFile = fastaFile + '.rcm'
    print("\nExtract reverse complement matching features successfully!\n")
    return rcmFile

if __name__ == '__main__':
    componentFeature("data/circRNA.seq")
    componentFeature("data/lncRNA.seq")
