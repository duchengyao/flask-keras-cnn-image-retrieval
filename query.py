# Author: S1NH.com

'''
Query

vec_multi:      普通KNN（矩阵相乘 + 结果排序）
faiss_flat:     Faiss 的 KNN (L2距离搜索)
faiss_ivfflat:  数据集分割成多个，在d维空间中定义Voronoi单元，每个数据库向量落在其中一个单元格中。
                在搜索时，只有查询x所在的单元格中包含的数据库向量y和几个相邻的数据库向量y与查询向量进行比较。
                    nlist:  单元格数
                    nprobe: 执行搜索访问的单元格数（nlist以外）
                    设置nprobe = nlist给出与faiss_flat搜索相同的结果（但较慢）
faiss_ivfpq:    为了扩展到非常大的数据集，基于乘积量化对数据进行压缩（有损压缩）
                    搜索真实查询时，虽然结果大多是错误的，但是它们在正确的空间区域，而对于真实数据，情况更好，因为：
                    1.统一数据很难进行索引，因为没有规律性可以被利用来聚集或降低维度
                    2. 对于自然数据，语义最近邻居往往比不相关的结果更接近。
'''

import numpy as np
import h5py
import config

METHOD = config.QUERY_METHOD
try:
    import faiss
except ImportError:
    METHOD = 'vec_multi'
    faiss = None

MAXRES = config.MAXRES
NLIST = 7
NPROBE = 10  # default nprobe is 1, try a few more


class Query:
    index_ivfflat = None
    imgNames = None
    feats = None
    quantizer = None

    def __init__(self, INDEX_FILE):
        # read in indexed images' feature vectors and corresponding image names
        h5f = h5py.File(INDEX_FILE, 'r')
        self.feats = h5f['dataset_1'][:]
        self.imgNames = h5f['dataset_2'][:]
        h5f.close()

        # # -----------------------------
        # index_flat = faiss.IndexFlatL2(self.feats[0].size)
        # index_flat.add(self.feats)
        # ------------------------------------
        nlist = NLIST
        self.quantizer = faiss.IndexFlatL2(self.feats[0].size)  # the other index
        self.index_ivfflat = faiss.IndexIVFFlat(self.quantizer, self.feats[0].size, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search
        assert not self.index_ivfflat.is_trained
        self.index_ivfflat.train(self.feats)
        assert self.index_ivfflat.is_trained

        self.index_ivfflat.add(self.feats)  # add may be a bit slower as well
        # ---------------------------------------

    def vec_multi(self, queryVec):
        scores = np.dot(queryVec[0], self.feats.T)
        rank_ID = np.argsort(scores)[::-1]
        # rank_score = scores[rank_ID]
        imlist = [self.imgNames[index] for i, index in enumerate(rank_ID[0:MAXRES])]
        return imlist

    def faiss_flat(self, queryVec):
        D, I = index_flat.search(queryVec, MAXRES)  # actual search
        imlist = [imgNames[index] for i, index in enumerate(I[0])]
        return imlist

    def faiss_ivfflat(self, queryVec):
        self.index_ivfflat.nprobe = NPROBE
        D, I = self.index_ivfflat.search(queryVec, MAXRES)
        imlist = [self.imgNames[index] for i, index in enumerate(I[0])]
        return imlist

    def action(self, queryVec):
        if METHOD is 'vec_multi':
            return self.vec_multi(queryVec)
        elif METHOD is 'faiss_flat':
            return self.faiss_flat(queryVec)
        elif METHOD is 'faiss_ivfflat':
            return self.faiss_ivfflat(queryVec)
        elif METHOD is 'faiss_ivfpq':
            pass


