#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import math
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

###  抽取主题中分值最大的句子编号，并更新矩阵  #########
def exLocalAndUpdate(sent_topicMatrix, topicN):
	INF = -100000000
	maxLocal = 0
	maxScore = INF
	#  获得编号
	for row in range(len(sent_topicMatrix)):
		if sent_topicMatrix[row][topicN] > maxScore:
			maxLocal = row
			maxScore = sent_topicMatrix[row][topicN]

	#  更新矩阵
	for col in range(len(sent_topicMatrix[0])):
		sent_topicMatrix[maxLocal][col] = INF

	return sent_topicMatrix, maxLocal

###  抽取主题中分值最大的三个关键词  #################
def exNKeyword(word_topicMatrix, topicN, wordList):
	keywordN = 3
	INF = -1000000
	keywordList = []

	for i in range(keywordN):
		maxLocal = 0
		maxScore = INF
		for row in range(len(word_topicMatrix)):
			if word_topicMatrix[row][topicN] > maxScore:
				maxScore = word_topicMatrix[row][topicN]
				maxLocal = row
		keywordList.append(wordList[maxLocal])
		word_topicMatrix[maxLocal][topicN] = INF

	return keywordList
 
###  判断句中包含的关键词，并更新关键词列表  #########
def getNewKeywordList(sentence, keywordList):
	new_keywordList = []
	for i in range(len(keywordList)):
		keyword = keywordList[i]
		if keyword not in sentence:
			new_keywordList.append(keyword)

	return new_keywordList


###  抽取摘要  ##########################################
def exAbstractModel(sentSegList, sentList):
	exsentList = []  # 抽取句子的结果

	vectorizer = CountVectorizer()  #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
	transformer = TfidfTransformer()  #该类会统计每个词语的tf-idf权值
	tfidf = transformer.fit_transform(vectorizer.fit_transform(sentSegList))  #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵 
	
	wordList = vectorizer.get_feature_names()  #获取词袋模型中的所有词语
	tfidfMatrix = tfidf.toarray()  #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
	TtfidfMatrix = tfidfMatrix.T  # 对tfidf矩阵进行转置

	U, S, V = np.linalg.svd(TtfidfMatrix)

	sentRate = 0.3  # 抽取句子的比例
	sentNum = len(sentList)  # 句子总数

	tempNum = float(sentNum) * sentRate
	exsentNum = int(sentNum * sentRate)  # 抽取的句子数
	if tempNum > exsentNum:
		exsentNum = exsentNum + 1

	# 句子抽取
	sentCount = 0
	exsentNumList = []
	topicN = 0 # 第几个主题
	sent_topicMatrix = V.T  # 句子-主题矩阵
	word_topicMatrix = U  # 词项-主题矩阵

	totalTopicSentNum = 3  # 每个主题下抽取句子的最大数量

	while sentCount < exsentNum:
		#
		sent_topicMatrix, maxLocal = exLocalAndUpdate(sent_topicMatrix, topicN)  # 抽取句子编号并更新矩阵
		exsentNumList.append(maxLocal)  # 存入抽取的句子编号
		sentCount = sentCount + 1
		#
		keywordList = exNKeyword(word_topicMatrix, topicN, wordList)  # 抽取关键词列表

		sentence = sentList[maxLocal]
		keywordList = getNewKeywordList(sentence, keywordList)  # 更新关键词列表
		#
		currentTopicSentNum = 1  # 当前主题下抽取出句子的数量
		while len(keywordList) > 0:
			if (currentTopicSentNum < totalTopicSentNum) and (sentCount < exsentNum):
				#
				sent_topicMatrix, maxLocal = exLocalAndUpdate(sent_topicMatrix, topicN)
				exsentNumList.append(maxLocal)
				sentCount = sentCount + 1
				currentTopicSentNum = currentTopicSentNum + 1
				#
				sentence = sentList[maxLocal]
				keywordList = getNewKeywordList(sentence, keywordList)

			else:
				break

		topicN = topicN + 1

	exsentResultList = sorted(exsentNumList)  # 对抽取的句子编号进行排序


	return exsentResultList

###  抽取摘要主函数  ##########################################################
def main():
	stateLocalName = 'ExtractTest/'
	fileReadPaperSegName = stateLocalName + 'part-00000-PaperSeg100'  # 分词文章
	fileReadPaperName = stateLocalName + 'part-00000-Paper100'  # 没分词文章
	fileReadPaperSeg = open(fileReadPaperSegName)
	fileReadPaper = open(fileReadPaperName)

	fileWriteResultName = fileReadPaperName + 'Result'
	fileWriteResult = open(fileWriteResultName, 'w')

	count = 0
	sentList = []
	sentSegList = []
	for line in fileReadPaperSeg.readlines():
		sourceLine = fileReadPaper.readline()
		if line == '-----------------------------------------------------------------\n':
				print '已经完成第', count, '个句子'
				if len(sentSegList) == 1:  # 如果文章中只有一个句子，不需要计算，直接输出
					out = '---------------------' + str(count) + '-------------------------\n'
					fileWriteResult.write(out)
					fileWriteResult.write(sentList[0])
					out = '-----------------------------------------------------------------\n'

					fileWriteResult.write(out)
					sentList = []
					sentSegList = []
					continue

				resultList = exAbstractModel(sentSegList, sentList)

				# 写结果
				out = '---------------------' + str(count) + '-------------------------\n'
				fileWriteResult.write(out)
				for i in range(len(resultList)):
					fileWriteResult.write(sentList[resultList[i]])
				out = '-----------------------------------------------------------------\n'
				fileWriteResult.write(out)

				sentList = []
				sentSegList = []
				continue
		elif line[: 8] == '--------':
			count = count + 1
			continue

		sentSegList.append(line)
		sentList.append(sourceLine)

	fileReadPaperSeg.close()
	fileReadPaper.close()
	fileWriteResult.close()

if __name__ == "__main__":
	main()
