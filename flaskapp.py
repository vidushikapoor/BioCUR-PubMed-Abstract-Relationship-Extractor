from flask import Flask
from flask import render_template
from flask import request
from flask import render_template_string
from flask import jsonify
import logging
from logging.handlers import RotatingFileHandler
app = Flask(__name__)

@app.route('/')
def show_form():
    return render_template("index.html")

@app.errorhandler(500)
def internal_error(error):
    return "custom 500 error"

@app.errorhandler(404)
def not_found(error):
    return "custom 404 error",404

@app.route('/result',methods=['POST','GET'])
def result():
    import os
    import re
    pmid = request.form['PubmedID']
    from Bio import Entrez
    from Bio.Entrez import efetch
    
    Entrez.email="vidushi421@gmail.com"
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml", rettype='abstract')
    a = handle.read()
    import xml.etree.cElementTree as etree
    tree = etree.XML(a)
    a=str()
    for node in tree.iter('AbstractText'):
        a=node.text
    abstract = a
    a=re.sub(r'[^\w]', ' ', a)
    sentence=str(a)
    writeList=[]
    relevantWords=[]
    app.logger.info("Abstract found from Entrez")
    response_dict = {}
    response_dict['pubmed_id'] = pmid
    if sentence:
     	response_dict["Sentence"] = sentence
    #response_dict['Current directory contents'] = os.listdir('.')#os.path.dirname(os.path.abspath(__file__))
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    nltk_words = open(cur_dir+'/static/nltk_words.txt').readlines()
    response_dict["NLTK words corpus loaded"] = True
    if nltk_words:
	response_dict["NLTK words"] = len(nltk_words)
    relevantWords = [item for item in sentence.split() if item not in nltk_words]
    app.logger.info("Relevant words: "+" ".join(relevantWords))
    uniqueRelevantWords=list(set(relevantWords))
    #struniqueRelevantWords=str(uniqueRelevantWords)
    struniqueRelevantWords = " ".join(uniqueRelevantWords)
    chunkList=['CHUNK:{<N.*> <N.*> <V.*> <N.*>}','CHUNK:{<N.*> <N.*> <V.*> <N.*><N.*>}','CHUNK:{<N.*> <V.*> <N.*><N.*>}','CHUNK:{<N.*> <N.*> <V.*> <N.*> <N.*> <N.*>}']
    import nltk
    nltk.data.path.append('/home/ubuntu/nltk_data/') 
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    stopWords = open(cur_dir+"/static/stop_words.txt").readlines()
    if struniqueRelevantWords:
	response_dict['Unique Relevant Words'] = struniqueRelevantWords
	response_dict['Stop Words length'] = len(stopWords)
    for i in chunkList:
        cp=nltk.RegexpParser(i)
        sent=sentence.split()
	sent = [word for word in sent if word not in stopWords]
        #for word in sent:
        #    if word in stopWords:
        #        sent.remove(word)
        sentence=str(sent)
        sentence=re.sub(r'[^\w]', ' ', sentence)
        tagged_sent=pos_tag(word_tokenize(sentence))
        tree=cp.parse(tagged_sent)
        for subtree in tree.subtrees():
            if subtree.label()=='CHUNK':
                strSubtree=str(subtree)
                temp1=strSubtree.replace('(',' ')
                temp2=temp1.replace(')','\n')
                temp3=temp2.replace('CHUNK',' ')
                temp4=re.sub(r'/[^\s]+','',temp3)
                for temp in temp4.split():
                    if temp in relevantWords:
                        writeList.append(temp4)
                        break
    writeList = [el.replace('\n', '') for el in writeList]
    if len(writeList)==0:
        return (' Ohh, their are no relevant biomedical relationships found in the abstract of given PubMed Id'+'.....'+'But we have relevant words for you'+'.....'+'Relevant Words::::' + str(uniqueRelevantWords))

    else:

        import random
        from collections import defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer
        import csv
        rel_list=[]
        with open(cur_dir+'/static/TrainingSet.csv','rb') as f:
            reader=csv.reader(f)
            for row in reader:
                rel_list.append(row)
	response_dict['Training set length'] = len(rel_list)

        def separateByClass(dataset):
            separated={}
            for i in range(len(dataset)):
                vector=dataset[i]
                vector[-1]=int(vector[-1])
                if(vector[-1]not in separated):
                    separated[vector[-1]]=[]
                separated[vector[-1]].append(vector)
            return separated

        separated=separateByClass(rel_list)
	response_dict["Number of items in class 0"] = len(separated[separated.keys()[0]])
	response_dict["Number of items in class 1"] = len(separated[separated.keys()[1]])
        relationList=[]
        labelList=[]
	
        for i in range(0,len(separated[-1])):
            relationList.append(separated[-1][i][0])
            labelList.append(separated[-1][i][1])
        for i in range(0,len(separated[1])):
            relationList.append(separated[1][i][0])
            labelList.append(separated[1][i][1])
	#if response_dict:
	#	return jsonify(**response_dict)
        from sklearn.feature_extraction import DictVectorizer
        from sklearn import svm
        def feature_extractor(data,stopWords):
            vectorizer=TfidfVectorizer(sublinear_tf=True, max_features = 3, max_df=0.5,stop_words=stopWords)
            mat = vectorizer.fit_transform(data)
            return mat

        def splitDataset(dataset,splitRatio):
            trainSize=int(len(dataset)*splitRatio)
            trainSet=[]
            copy=list(dataset)
            while len(trainSet)<trainSize:
                index=random.randrange(len(copy))
                trainSet.append(Copy.pop(index))
            return [trainSet,copy]

        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import ShuffleSplit

	response_dict['Length of relations and labels list'] = len(relationList)
	#if response_dict:
	#	return jsonify(**response_dict)
        X=feature_extractor(relationList,stopWords)
        Y=labelList
        scoreList=[]
        for i in range(1,10):
            train,test,train_label,test_label=train_test_split(X,Y,test_size=0.40)
            clf=svm.SVC()
            clf.fit(train,train_label)
            scoreList.append(clf.score(test,test_label))
        predicted=clf.predict(test)
	
# Naive Bayes ////////////////////////////////////////////////////////////////////////////////

        from sklearn.naive_bayes import GaussianNB

#Tokenizing text with scikit-learn
        from sklearn.feature_extraction.text import CountVectorizer
        count_vect=CountVectorizer()
        X_train_counts=count_vect.fit_transform(relationList)

#From occurrences to frequencies
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Training a classifier
        from sklearn.naive_bayes import MultinomialNB
        clf=MultinomialNB().fit(X_train_tfidf,labelList)

#Building a pipeline
        from sklearn.pipeline import Pipeline

        text_clf=Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
        naiveScoreList=[]
        for i in range(1,10):
            train,test,train_label,test_label=train_test_split(relationList,labelList,test_size=0.40)
            text_clf=text_clf.fit(train, train_label)
            text_clf.predict(test)
            naiveScoreList.append(text_clf.score(test,test_label))

###### Test Case

        dic={}
        predicted=text_clf.predict(writeList)
        for i in range(0,len(writeList)):
            dic[writeList[i]]=predicted[i]
	response_dict['Result Words'] = dic
	#if response_dict:
	#	return render_template("results.html",abstract=abstract,relationships=)
		#return jsonify(**dic)
        #return str(dic)
	return render_template("results.html",abstract=abstract,relationships=dic)	

if __name__ == '__main__':
	handler = RotatingFileHandler('/home/ubuntu/flaskapp/foo.log',maxBytes=10000,backupCount=1)
	handler.setLevel(logging.INFO)
	app.logger.addHandler(handler)
	app.debug=True
	app.run()
