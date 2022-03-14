import numpy as np
import pandas as pd
import math

import simplejson as json
import re

import pycountry
import networkx as nx
from copy import copy
import os


class commands:

    def help(self):
        print("Very comprehensive help!")

    def getDF(self,fileType="csv"):

        postfix = 0
        if fileType == "excel":
            fileType = ".xls"
        else:
            fileType = ".csv"
        
        fName = "./WoSExported0"+str(fileType)
        try:
            df = pd.read_excel(fName)
        except:
            print("There is no csv or excel file for starting the parsing ...")

        while True:
            fName = "./WoSExported"+str(postfix)+str(fileType)
            if os.path.isfile(fName):
                df1 = pd.read_excel(fName)
                df = pd.concat([df, df1], ignore_index=True)
            else:
                break
            postfix += 1
        
        df = df.drop_duplicates()
        return df

    
    def getTheJSON(self,path='./data.json'):
        
        df = self.getDF(fileType="excel")
        
        # Convert useful data to json
        papers = {}

        topicVector = ["Machine Learning","Deep Learning","Reinforcement Learning"," ML "," RL ","Neural Network"," CNN ","Convolutional Neural Network"," LSTM ","Long Short-term Memory",
                    " GAN ","Generative adversarial networks","Decision Tree","Random Forest"," SVM "," SVC ","Regression","Unsupervised Learning ","Supervised Learning"," PCA "," K-Means ",
                    " KNN ","Logistic Regression"," RNN ","Recurrent Neural Network","Computer Vision","Machine Vision","Pattern Recognition"," VR "," AR ","Augmented Reality","Virtual Reality",
                    "Digital Twin","Blockchain","5G","Metaheuristic Algorithms","Meta-heuristic algorithms","Metaheuristic"," PSO ","Genetic Algorithm","Ant Colony","Classification",
                    ]

        topicDict = {"Machine Learning":"Machine Learning","Deep Learning":"Deep Learning",
                    "Reinforcement Learning":"Reinforcement Learning"," ML ":"Machine Learning"," RL ":"Reinforcement Learning",
                    " CNN ":"Computer Vision","Convolutional Neural Network":"Computer Vision",
                    " LSTM ":"Deep Learning","Long Short-term Memory":"Deep Learning",
                    " GAN ":"Deep Learning","Generative adversarial networks":"Deep Learning","Decision Tree":"Machine Learning",
                    "Random Forest":"Machine Learning"," SVM ":"Machine Learning"," SVC ":"Machine Learning","Regression":"Machine Learning",
                    "Unsupervised Learning ":"Machine Learning","Supervised Learning":"Machine Learning",
                    " PCA ":"Machine Learning"," K-Means ":"Machine Learning",
                    " KNN ":"Machine Learning","Logistic Regression":"Machine Learning"," RNN ":"Deep Learning",
                    "Recurrent Neural Network":"Deep Learning","Computer Vision":"Computer Vision","Machine Vision":"Computer Vision","Pattern Recognition":"Deep Learning",
                    " VR ":"Virtual Reality"," AR ":"Augmented Reality","Augmented Reality":"Augmented Reality",
                    "Virtual Reality":"Virtual Reality","Ethereum":"Blockchain","Solidity":"Blockchain",
                    "Digital Twin":"Digital Twin","Blockchain":"Blockchain","5G":"5G","Metaheuristic Algorithms":"Meta-Heuristic Algorithms",
                    "Meta-Heuristic Algorithms":"Meta-Heuristic Algorithms","Metaheuristic":"Meta-Heuristic Algorithms",
                    " PSO ":"Meta-Heuristic Algorithms","Genetic Algorithm":"Meta-Heuristic Algorithms","Ant Colony":"Meta-Heuristic Algorithms",
                    "Particle Swarm Optimization":"Meta-Heuristic Algorithms","Classification":"Machine Learning","5th generation mobile network":"5G",
                    "Support Vector Machine":"Machine Learning","Naive Bayes":"Machine Learning","Hierarchical Clustering":"Machine Learning","Stochastic Gradient Descent":"Machine Learning",
                    "auto encoder":"Deep Learning","auto-encoder":"Deep Learning","auto-encoders":"Deep Learning","AdaBoost":"Machine Learning",
                    "Bag of Words":"NLP","NLP":"NLP","word to vec":"NLP","Natural Language Processing":"NLP",
                    "wordtovec":"NLP","Sentiment Analysis":"NLP","tf-idf":"NLP","Language Processing":"NLP","Policy Gradient":"Reinforcement Learning",
                    "Q-Learning":"Reinforcement Learning","SARSA":"Reinforcement Learning","DDGP":"Reinforcement Learning","Actor-Critic":"Reinforcement Learning","DQN":"Reinforcement Learning"}

        # topicDict = { "Deep Learning":"Deep Learning","Reinforcement Learning":"Reinforcement Learning",
        #              "Imitation learning":"Imitation Learning","Learning by Demonstration":"Imitation Learning","Sampling-Based Methods":"Sampling-Based Methods",
        #              "Sampling-Based Motion Planners":"Sampling-Based Methods"," RRT ":"Sampling-Based Methods",
        #              " RRT* ":"Sampling-Based Methods"," PRM ":"Sampling-Based Methods","sampling-based":"Sampling-Based Methods","Metaheuristic Algorithms":"Bio-Inspired Heuristic",
        #              "Meta-Heuristic Algorithms":"Bio-Inspired Heuristic","Metaheuristic":"Bio-Inspired Heuristic",
        #              " PSO ":"Bio-Inspired Heuristic","Genetic Algorithm":"Bio-Inspired Heuristic","Ant Colony":"Bio-Inspired Heuristic",
        #              "Particle Swarm Optimization":"Bio-Inspired Heuristic","Potential Field":"Potential Field"," LSTM ":"Deep Learning"," GAN ":"Deep Learning",
        #              "Decision Tree":"Deep Learning"," CNN ":"Deep Learning", " RNN ":"Deep Learning","Learning from Demonstration":"Imitation Learning",
        #              " auto encoder ":"Deep Learning", "Sampling-Based Methods":"Sampling-Based Methods",  "random rapidly exploring tree":"Sampling-Based Methods"  }


        functions = {}
        functions["resourceAllocation"] = ["allocating the resources", "allocation of resources", "resource allocation", "resource allocating", "resource planning", "plarn resources", "planning the resources"]
        functions["operationsScheduling"] = ["scheduling the operations", "operation scheduling", "operation schedule", "schedule the tasks", "task scheduling", "scheduling system"]
        functions["dispatchingProductionUnits"] = ["Dispatch product", "dispatching the product", "product dispatching", "Dispatch material", "dispatching the material", "material dispatching",]
        functions["documentControl"] = ["document control", "smart document", "controlling the document", "automated document"]
        functions["dataCollection"] = ["Data collection", "data gathering", "data acquisition", "collecting the data", "acquiring the data", "collect the data", "acquire the data"]
        functions["labourManagement"] = ["Labour management", "managing labour", "Shift management", "managing shift", "staff management", "Managing the staff"]
        functions["qualityManagement"] = ["Quality control", "control the quality", "quality management", "quality assessment", "assessing the quality", "assess the quality", "Quality improvement"]
        functions["processManagement"] = ["Process management", "Managing the process", "manage the process", "process control", "control the process"]
        functions["maintenanceManagement"] = ["maintenance"]
        functions["productTracking"] = ["Product tracking", "tracking the product", "product traceability", "track the product"]
        functions["performanceAnalysis"] = ["performance analysis", "performance assessment", "assessing the performance", "measuring the performance"]

        con = 0

        funcDistribution = {}
        for i, row in df.iterrows(): # For each paper
            # Extract affiliations/countries from addresses
            affiliations = {}
            if isinstance(row['Addresses'], str):
                affiliation_addresses = re.sub("[\(\[].*?[\)\]]", "", row['Addresses']).split(';')
                for affiliation in affiliation_addresses:
                    temp = affiliation.split(',')
                    if any(map(str.isdigit, temp[-1])) or ' USA' in temp[-1].upper(): # Making USA different names union
                        affiliations[temp[0].strip(' ')] = 'USA'
                    else:
                        affiliations[temp[0].strip(' ')] = temp[-1].strip(' ')
            affiliationSet = list(set(affiliations.keys()))
            countriesSet = list(set(affiliations.values()))


            # Extract List of Authors
            authors = []
            if isinstance(row['Author Full Names'], str):
                for author in row['Author Full Names'].split(";"):
                    authors.append(author.strip().title())
            authors = list(set(authors))

            # Extract List of Keywords
            keywords = []
            if isinstance(row['Author Keywords'], str):
                for word in row['Author Keywords'].split(";"):
                    keywords.append(word.strip().title())

            # Extract topic based vectors
            methodsScore = {}

            threshold = 1.1
            abstractCoefficient = 1
            keywordCoefficient = 2
            titleCoefficient = 3


            for method in topicVector:
                tempScore = 0
                
                tempScore = titleCoefficient*int(str(row["Article Title"]).lower().count(method.lower())) + keywordCoefficient*int(str(row["Author Keywords"]).lower().count(method.lower())) + abstractCoefficient*int(str(row["Abstract"]).lower().count(method.lower()))

                if tempScore>threshold:
                    methodsScore[method] = tempScore
            
            # Extract general topic based vectors
            areaFrequency = {}
            for method,area in topicDict.items():
                tempScore = 0
                
                tempScore = 3*int(str(row["Article Title"]).lower().count(method.lower())) + 2*int(str(row["Author Keywords"]).lower().count(method.lower())) + 1*int(str(row["Abstract"]).lower().count(method.lower()))

                if tempScore>1:
                    areaFrequency[area] = 1
            
            # Extract MES functions of each paper
            funcScores = {}
            
            for func,vect in functions.items():
                tempScore = 0
                for method in vect:
                    tempScore = tempScore + (3*int(str(row["Article Title"]).lower().count(method.lower())) + 0*int(str(row["Author Keywords"]).lower().count(method.lower())) + 0*int(str(row["Abstract"]).lower().count(method.lower())))
                
                funcScores[func] = tempScore
                if tempScore > 0 :
                    if func in funcDistribution:
                        funcDistribution[func] = funcDistribution[func] + 1
                    else:
                        funcDistribution[func] = 1

            con += 1
            papers[row['UT (Unique WOS ID)']] = {
                'title': row['Article Title'].title(),
                'abstract': row['Abstract'],
                'source_title': row['Source Title'],
                'publisher': row['Publisher'],
                'page_count': row['Number of Pages'],
                'year': row['Publication Year'],
                'cited_references': row['Cited Reference Count'],
                'times_cited_wos': row['Times Cited, WoS Core'],
                'times_cited_all': row['Times Cited, All Databases'],
                'access_six_months': row['180 Day Usage Count'],
                'affiliations': affiliations,
                'affiliationSet': affiliationSet,
                'countries': countriesSet,
                'authors': authors,
                'keywords': keywords,
                'functions': funcScores,
                'areas': areaFrequency,
                'methods': methodsScore
            }

        with open(path,'w') as outfile:
            json.dump(papers, outfile, ignore_nan=True)