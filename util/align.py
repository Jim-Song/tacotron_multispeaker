import numpy as np
#from hparams import hparams



def enlarge(alignment, n=1):
    alignment2 = []
    for tag in alignment:
        crrt_line = []
        for i in range(0, len(tag)-1):
            a = tag[i]
            b = tag[i + 1]
            for j in range(1, n+1):
                tmp = a*(n+1-j)/(n+1) + b*j/(n+1)
                crrt_line.append(tmp)
        alignment2.append(crrt_line)
    return np.asarray(alignment2)

#a = [[1,2,3,4,5],[6,7,8,9,0]]
#print(enlarge(a,2))












