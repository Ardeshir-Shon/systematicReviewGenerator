# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math
!pip install simplejson
import simplejson as json
import re
!pip install pycountry
import pycountry
import networkx as nx
from copy import copy
!pip install xlrd
!pip install --upgrade xlrd

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

# %cd drive/MyDrive/DataForColab

df1 = pd.read_excel("mupdatedQuery1.xls")
# df2 = pd.read_excel("m2updatedQuery2.xls")
# df3 = pd.read_excel("m2updatedQuery3.xls")
# df4 = pd.read_excel("m2updatedQuery4.xls")
# df5 = pd.read_excel("m2updatedQuery5.xls")
df = pd.concat([df1], ignore_index=True)
print(df)

gdf = df[["Source Title","Article Title"]].groupby(["Source Title"]).count().sort_values(by="Article Title",ascending=False).head(15)
ydf = df[df["Publication Year"]>2008]
gdf = gdf.join(ydf[["Publication Year","Source Title","UT (Unique WOS ID)"]].set_index("Source Title")).reset_index()
gdf.groupby(["Source Title","Publication Year"]).count().to_excel("journals.xlsx")

len(df)

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

with open('data.json', 'w') as outfile:
    json.dump(papers, outfile, ignore_nan=True)

trends = dict((k,list(np.zeros(14))) for k in list(set(topicDict.values())))

nans = 0
print(trends)
loopCount = 0
overalPapers = list(np.zeros(14))
overalAdded = 0
for DOI,info in papers.items():
  loopCount += 1
  year = info['year']
  if math.isnan(year) or year < 2008 or year > 2021:
    nans = nans + 1
    continue
  areas = info['areas']
  for topic,score in areas.items():
    yIndex = int(year)-2008
    trends[topic][yIndex] += 1
    overalPapers[yIndex] += 1
    overalAdded += 1

normalizedTrends = copy(trends)
hotTopicNess ={}
topicIdf = {}
topicTf = {}

print(overalPapers)

for topic,l in trends.items():
  # normalizedTrends[topic] = list(np.array(trends[topic])/np.array(overalPapers))
  iter = 0
  hotTopicNess[topic] = []
  topicIdf[topic] = []
  topicTf[topic] = []
  
  for freq in l:
    try:
      if iter > 2:
        idf = math.log(sum(overalPapers[iter-3:iter])/float(1+sum(l[iter-3:iter])),1.3)
      else:
        idf = math.log(sum(overalPapers[:iter])/float(1+sum(l[:iter])),1.3)
    except:
      idf = 1
    if idf == 0:
      if iter > 2:
        idf = math.log(sum(overalPapers[iter-3:iter])/float(sum(l[iter-3:iter])),1.3)
      else:
        idf = math.log(sum(overalPapers[:iter])/float(sum(l[:iter])),1.3)
    
    hotTopicNess[topic].append(l[iter]*pow(idf,1.5))
    topicIdf[topic].append(idf)
    topicTf[topic].append(l[iter])
    iter += 1

# print("IDF: ",topicIdf["Machine Learning"])
# print("TF: ",topicTf["Machine Learning"])
# print("hotness: ",hotTopicNess["Machine Learning"])
# print("=========================")
# # print("IDF: ",topicIdf["Blockchain"])
# # print("TF: ",topicTf["Blockchain"])
# print("hotness: ",hotTopicNess["Blockchain"])
# print("============================")
# print("hotness: ",hotTopicNess["Reinforcement Learning"])
# print("============================")
# print("hotness: ",hotTopicNess["Digital Twin"])
# print("============================")
# hotTopicNess

trends

pick1 = []
pick2 = []
pick3 = []
pick4 = []

topic1 = []
topic2 = []
topic3 = []
topic4 = []

for yearIndex in range(14):
  p1 = 0
  p2 = 0
  p3 = 0
  p4 = 0

  t1 = ""
  t2 = ""
  t3 = ""
  t4 = ""

  for topic in set(topicDict.values()):

    if hotTopicNess[topic][yearIndex] > p1:
      p4 = p3
      t4 = t3

      p3 = p2
      t3 = t2

      p2 = p1
      t2 = t1
      
      p1 = hotTopicNess[topic][yearIndex]
      t1 = topic

    elif hotTopicNess[topic][yearIndex] > p2:
      p4 = p3
      t4 = t3

      p3 = p2
      t3 = t2
      
      p2 = hotTopicNess[topic][yearIndex]
      t2 = topic

    elif hotTopicNess[topic][yearIndex] > p3:
      p4 = p3
      t4 = t3
      
      p3 = hotTopicNess[topic][yearIndex]
      t3 = topic

    elif hotTopicNess[topic][yearIndex] > p4:
      p4 = hotTopicNess[topic][yearIndex]
      t4 = topic
    
  pick1.append(p1)
  pick2.append(p2)
  pick3.append(p3)
  pick4.append(p4)

  topic1.append(t1)
  topic2.append(t2)
  topic3.append(t3)
  topic4.append(t4)


print(pick1)
print(pick2)
print(pick3)
print(pick4)

print(topic1)
print(topic2)
print(topic3)
print(topic4)

import matplotlib.pyplot as plt
import statistics

plt.xlabel('Year')
plt.ylabel('Frequency')
yearsList = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]

for topic,trend in trends.items():
  plt.plot(yearsList,trend,'-o',label=topic)

plt.legend()
# plt.show()
plt.savefig("topicTrends.png",transparent=True,dpi=300)

trends.keys()

import matplotlib.pyplot as plt
import statistics

plt.xlabel('Year')
plt.ylabel('Frequency')
yearsList = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]

plt.plot(yearsList,trends['Digital Twin'],marker='$DT$',label=topic)

plt.legend()
plt.show()
# plt.savefig("topicTrends.png",transparent=True)

myDict = {
    2009:"Machine Learning",
    2010:"Virtual Reality",
    2011:"Machine Vision",
    2012:"Machine Learning",
    2013:"Pattern Recognition",
    2014:"Machine Vision",
    2015:"Genetic Algorithm",
    2016:"Augmented Reality",
    2017:"Deep Learning",
    2018:"Digital Twin",
    2019:"Reinforcement Learning",
    2020:"5G",
    2021:"Blockchain"
}

result = {}
year = 2009
keyword = "Machine Learning"
for year,keyword in myDict.items():
  maxNo = 0
  maxDOI = '' 
  for DOI,paper in papers.items():
    if math.isnan(paper['year'])==False and int(paper['year'])==year and keyword in paper['methods']:
      if paper['times_cited_all']>=maxNo:
        maxNo = paper['times_cited_all']
        maxDOI = DOI
      elif keyword == "Augmented Reality":
        print(paper)
        
  result[year] = papers[maxDOI]
with open('perYear.json', 'w') as outfile:
    json.dump(result, outfile, ignore_nan=True)

keywordYear = {}
# years = ['2009','2010','2011','2012','2013','2014','2015','2015','2016','2017','2018','2019','2020','2021']
# for year in years:
for k,v in papers.items():
  temp = math.isnan(v['year'])
  if temp == True:
    continue
  year = int(v['year'])
  if year in keywordYear:
    for key,value in v['methods'].items():
      if key in keywordYear[year]:
        # keywordYear[year][key] = keywordYear[year][key] + value
        if value>1:
          keywordYear[year][key] = keywordYear[year][key] + 1
      else:
        # keywordYear[year][key] = value
        if value>1:
          keywordYear[year][key] = 1
  else:
    temp = {}
    for key,value in v['methods'].items():
      temp[key]=1
    # keywordYear[year] = v['methods']
    keywordYear[year] = temp

keywordYear
sortedKeywords = {}
for year,value in keywordYear.items():
  sortedKeywords[year] = {k: v for k, v in sorted(value.items(), reverse= True, key=lambda item: item[1])}
sortedKeywords

authorsContribution = {}
authorsCitation = {}
for k,v in papers.items():
  for author in v['authors']:
    if author in authorsContribution:
      authorsContribution[author] = authorsContribution[author]+1
      authorsCitation[author] = authorsCitation[author] + v['times_cited_all']
    else:
      authorsContribution[author] = 1
      authorsCitation[author] = v['times_cited_all']
authorsContribution = {k: v for k, v in sorted(authorsContribution.items(), reverse= True, key=lambda item: item[1])}
authorsCitation = {k: v for k, v in sorted(authorsCitation.items(), reverse= True, key=lambda item: item[1])}

affiliationContribution = {}
affiliationCitation = {}
for k,v in papers.items():
  for affiliation in v['affiliationSet']:
    if affiliation in affiliationContribution:
      affiliationContribution[affiliation] = affiliationContribution[affiliation]+1
      affiliationCitation[affiliation] = affiliationCitation[affiliation] + v['times_cited_all']
    else:
      affiliationContribution[affiliation] = 1
      affiliationCitation[affiliation] = v['times_cited_all']
affiliationContribution = {k: v for k, v in sorted(affiliationContribution.items(), reverse= True, key=lambda item: item[1])}
affiliationCitation = {k: v for k, v in sorted(affiliationCitation.items(), reverse= True, key=lambda item: item[1])}

countriesContribution = {}
countriesCitation = {}
for k,v in papers.items():
  for country in v['countries']:
    if country in countriesContribution:
      countriesContribution[country] = countriesContribution[country]+1
      countriesCitation[country] = countriesCitation[country] + v['times_cited_all']
    else:
      countriesContribution[country] = 1
      countriesCitation[country] = v['times_cited_all']
countriesContribution = {k: v for k, v in sorted(countriesContribution.items(), reverse= True, key=lambda item: item[1])}
countriesCitation = {k: v for k, v in sorted(countriesCitation.items(), reverse= True, key=lambda item: item[1])}

countriesContribution
with open('countTemp.csv', 'w') as f:
  for key in countriesContribution.keys():
    f.write("%s,%s\n"%(key,countriesContribution[key]))

affiliationContribution
with open('affTemp.csv', 'w') as f:
  for key in affiliationContribution.keys():
    f.write("%s,%s\n"%(key,affiliationContribution[key]))

newDf = df[["UT (Unique WOS ID)","Publication Year","Times Cited, All Databases"]]
newDf["Publication Year"] = pd.to_numeric(newDf["Publication Year"],downcast='integer')

groupedDf = newDf.groupby(["Publication Year"]).sum()


# enable i10 in next line
newDf = newDf[newDf["Times Cited, All Databases"]>=10]


# groupedDf = newDf.groupby(["Publication Year"]).sum()
groupedDf["count"] = newDf.groupby(["Publication Year"]).count()["UT (Unique WOS ID)"]
groupedDf.dropna(inplace = True)
groupedDf

GDP = pd.read_excel('world.xls')
output = GDP.set_index(["Year"]).join(groupedDf).dropna()

# output

print(output.corr())

from scipy.stats import pearsonr

correlation, p_value = pearsonr(output['GDP'], output['Times Cited, All Databases'])

print(correlation , p_value)

correlation, p_value = pearsonr(output['GDP'], output['count'])

print(correlation , p_value)

# import the library
import matplotlib.pyplot as plt
import statistics

# output = output.reset_index().reset_index('index')

# Creation of Data
x1 = output.reset_index()["Year"]#.reset_index()['index']
y1 = (output['GDP']-statistics.mean(output['GDP']))/(statistics.stdev(output['GDP']))
y2 = (output['count']-statistics.mean(output['count']))/(statistics.stdev(output['count']))
y3 = (output['Times Cited, All Databases']-statistics.mean(output['Times Cited, All Databases']))/(statistics.stdev(output['Times Cited, All Databases']))
# y4 = y3/y2

# Plotting the Data
plt.plot(x1, y1, '--', label='World GDP')
plt.plot(x1, y2 , 'o', label='Number of i-10 Papers')
plt.plot(x1, y3, label='Sum of the Citations')
# plt.plot(x1, y4, label='Mean of citation per paper')

plt.xlabel('Year')
plt.ylabel('Z-Score Scaled Value')

plt.legend()
# plt.show()
plt.savefig("WCCorr.png",transparent=True,dpi = 300)

# correlation, p_value = stats.pearsonr(x, y)
# countriesCitation
dfCountries = pd.DataFrame(columns=('country', 'citation','contribution'))
i = 0
for country,cite in countriesCitation.items():
  dfCountries.loc[i] = [country, cite, 0]
  i = i+1
i = 0
for country,contribution in countriesContribution.items():
  dfCountries.loc[i] = [country, dfCountries.loc[i][1] , contribution]
  i = i+1
dfCountries.head(30)

GDP = pd.read_excel('GDP.xls')
output = GDP.set_index('country').join(dfCountries.set_index('country')).dropna()

from scipy.stats import pearsonr

correlation, p_value = pearsonr(output['GDP'], output['contribution'])

correlation , p_value
# output.reset_index().reset_index()

# import the library
import matplotlib.pyplot as plt
import statistics

# output = output.reset_index().reset_index('index')

# Creation of Data
x1 = output.reset_index().reset_index()['index']
y1 = (output['GDP']-statistics.mean(output['GDP']))/(statistics.stdev(output['GDP']))
y2 = (output['citation']-statistics.mean(output['citation']))/(statistics.stdev(output['citation']))

# Plotting the Data
plt.plot(x1, y1, '-o' , label='Countries GDP of 2020')
plt.plot(x1, y2, '-o', label='Contribution in the Intelligent-MES ')

plt.xlabel('Country Ids')
plt.ylabel('Z-Score Scaled Value')

plt.legend()
plt.savefig("CGDPCorr.png",transparent=True,dpi=300)

G = nx.Graph()

for k,v in papers.items():
  for country in v['countries']:
    for country2 in v['countries']:
      if country == country2:
        continue
      if G.has_edge(country,country2)==True:
        weight = G.get_edge_data(country, country2)
        G[country][country2]['weight'] = weight['weight']+1
      else:
        G.add_edge(country,country2,weight=1)
nx.write_gexf(G, "countries.gexf")

edges=sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1),reverse=True)
edges

G = nx.Graph()

for k,v in papers.items():
  for affiliation in v['affiliationSet']:
    for affiliation2 in v['affiliationSet']:
      if affiliation == affiliation2:
        continue
      if G.has_edge(affiliation,affiliation2)==True:
        weight = G.get_edge_data(affiliation, affiliation2)
        G[affiliation][affiliation2]['weight'] = weight['weight']+1
      else:
        G.add_edge(affiliation,affiliation2,weight=1)
nx.write_gexf(G, "affiliations.gexf")

edges=sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1),reverse=True)
edges

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

print(sum(value >= 2 for key,value in authorsContribution.items()))
print(np.median(list(authorsContribution.values())))
print(len(authorsContribution.keys()))
n_authors = take(16, authorsContribution.items())
n_authors

print(sum(value >= 200 for key,value in authorsCitation.items()))
print(np.median(list(authorsCitation.values())))
print(len(authorsCitation.keys()))
n_authors = take(16, authorsCitation.items())
n_authors

print(sum(value >= 5 for key,value in affiliationContribution.items()))
print(np.median(list(affiliationContribution.values())))
print(len(affiliationContribution.keys()))
n_authors = take(16, affiliationContribution.items())
n_authors

print(sum(value >= 200 for key,value in affiliationCitation.items()))
print(np.median(list(affiliationCitation.values())))
print(len(affiliationCitation.keys()))
n_authors = take(16, affiliationCitation.items())
n_authors

print(sum(value >= 30 for key,value in countriesContribution.items()))
print(np.median(list(countriesContribution.values())))
print(len(countriesContribution.keys()))
n_authors = take(16, countriesContribution.items())
n_authors

countriesContribution

with open('countries.tsv', 'w') as the_file:
    the_file.write('Country\tPaper Contribution\tContribution Share\n')
    for country,paper in countriesContribution.items():
      the_file.write(country)
      the_file.write('\t')
      the_file.write(str(paper))
      the_file.write('\t')
      the_file.write(str(round(paper/1383*100,2)))
      the_file.write('\n')

len(affiliationContribution.keys())
# sum([0 if int(cont) > 1 else -1 for cont in authorsContribution.values()])

!pip install pycountry
import pycountry
text = "United States (New York), United Kingdom (London)"
for country in pycountry.countries:
    if country.name in text:
        print(country.name)

def computeTF(query, document):
    
    global occurenceGlob
    
    documentWords = document.split(' ')
    queryWords = query.split(',')
    documentLength = len(documentWords)
    uniqueQueryWords = set(queryWords)
    intersection = set(queryWords).intersection(set(documentWords))
    intersection = dict.fromkeys(intersection, 0)
    for word,occurence in intersection.items():
      for w in documentWords:
        if w == word:
          intersection[word] += 1
          occurenceGlob[word] += 1

    for word,occurence in intersection.items():
      intersection[word] =  occurence/documentLength

    return intersection

def computeIDF(query,documents):
    
    global idfGlob
    
    import math
    N = len(documents)

    queryWords = query.split(',')
    idfs = dict.fromkeys(set(queryWords), 1)

    for word,idf in idfs.items():
        if(idfGlob[word]==0):
            for document in documents:
                tempstr = " "+str(word)+" "
                if tempstr in document:
                    idfs[word] += 1
        else:
            continue
    for word, val in idfs.items():
        if(idfGlob[word]==0):
            idfs[word] = math.log( (N+1) / float(val))
            idfGlob[word] = idfs[word]
        else:
            idfs[word] = idfGlob[word]
    return idfs

def computeTFIDF(tfs,idfs):
    tfidf = dict.fromkeys(set(tfs.keys()), 0)

    for word,tf in tfs.items():
        tfidf[word] = tf*idfs[word]
    
    return tfidf

df = pd.read_excel("Query1.xls")
df = df[["DOI","Article Title","Author Keywords","Keywords Plus","Abstract"]].dropna(subset=["DOI"])
df

topicVector = ["Machine Learning","Deep Learning","Reinforcement Learning"," ML "," RL ","Neural Network"," CNN ","Convolutional Neural Network"," LSTM ","Long Short-term Memory",
               " GAN ","Generative adversarial networks","Decision Tree","Random Forest"," SVM "," SVC ","Regression","Unsupervised Learning ","Supervised Learning"," PCA "," K-Means ",
               " KNN ","Logistic Regression"," RNN ","Recurrent Neural Network"] 
for method in topicVector:
  print(method)

idfGlob = defaultdict(float)
occurenceGlob = defaultdict(int)
# processed_tweets = processed_tweets[0:1000]

# queries = ["???????? ????????,??????,????????,??????????,?????????? ????????,????????????,???????????? ????????????,????????????,??????,????????,????????????,??????????,????????????,????????????,??????????,??????,????????,????????,??????,??????????,????????,????????????,??????????????"
#            ,"??????????,????????????,??????????????,?????????????? ????????????,??????????,????????????,????????,????????????,???????? ?????? ????,????????????"
# ,"????,???????? ????,???? ????,??????????????,?????????? ????,????????????,??????????,???????? ??????,??????????????????,??????,??????????,??????????,??????????,????????????????????????,???? ????????,???? ??????????,????????????????,??????????????????,??????????,??????????,???????????? ????????????,??????????"
#           ,"?????????? ????????????,????????,???????? ????????,??????????,?????????? ??????????,????????????????,????????????????????,?????????? ????????,?????????? ??????????,????????,?????? ??????????,??????????,????????????,??????????????,??????????,????????,????????,?????????? ??????????????,????????,????,??????????,????????,??????,????????"]


lables = ["??????????????","????????????","??????????","???????????? ?? ??????????????","???????? ?????????? ???? ????????"]
output = pd.DataFrame(" ", index=list(range(len(processed_tweets))), columns=['Tweet','Tweet_id','Likes','Retweets','Time','Time_Id','FirstVector','SecondVector','ThirdVector','FourthVector','Matched','Lable'])

documentId = 0

for document in processed_tweets:
  if documentId%500 == 0:
    print("#document:",documentId)
  queryId = 0
  for query in queries:
    # print("query:",queryId )
    output.at[documentId,'Tweet'] = document
    output.at[documentId,'Tweet_id'] = tweet_ids[documentId]
    output.at[documentId,'Likes'] = likes[documentId]
    output.at[documentId,'Retweets'] = retweets[documentId]
    output.at[documentId,'Time'] = times[documentId]
    output.at[documentId,'Time_Id'] = time_ids[documentId]

    if queryId ==0:
      tfs=computeTF(query,document)
      idfs=computeIDF(query,processed_tweets)
      firstTFIDF = sum(dict(computeTFIDF(tfs,idfs)).values())
      # firstTFIDF = sum(dict(computeTFIDF(computeTF(query,document),computeIDF(query,processed_tweets)).values()))
      output.at[documentId ,'FirstVector'] = firstTFIDF
    
    elif queryId ==1:
      tfs=computeTF(query,document)
      idfs=computeIDF(query,processed_tweets)
      secondTFIDF = sum(dict(computeTFIDF(tfs,idfs)).values())
      # secondTFIDF = sum(dict(computeTFIDF(computeTF(query,document),computeIDF(query,processed_tweets)).values()))
      output.at[documentId , 'SecondVector'] = secondTFIDF
    
    elif queryId ==2:
      tfs=computeTF(query,document)
      idfs=computeIDF(query,processed_tweets)
      thirdTFIDF = sum(dict(computeTFIDF(tfs,idfs)).values())
      # secondTFIDF = sum(dict(computeTFIDF(computeTF(query,document),computeIDF(query,processed_tweets)).values()))
      output.at[documentId , 'ThirdVector'] = thirdTFIDF
    
    elif queryId ==3:
      tfs=computeTF(query,document)
      idfs=computeIDF(query,processed_tweets)
      fourthTFIDF = sum(dict(computeTFIDF(tfs,idfs)).values())
      # thirdTFIDF = sum(dict(computeTFIDF(computeTF(query,document),computeIDF(query,processed_tweets)).values()))
      output.at[documentId , 'FourthVector'] = fourthTFIDF

      if (firstTFIDF == secondTFIDF) and (firstTFIDF == thirdTFIDF) and (firstTFIDF == fourthTFIDF) :
        output.at[documentId , 'Matched'] = 5 # none of them 
      elif  (firstTFIDF >= secondTFIDF) and (firstTFIDF >= thirdTFIDF) and (firstTFIDF >= fourthTFIDF):
        output.at[documentId , 'Matched'] = 1
      elif  (firstTFIDF <= secondTFIDF) and (secondTFIDF >= thirdTFIDF) and (secondTFIDF >= fourthTFIDF):
        output.at[documentId , 'Matched'] = 2
      elif  (thirdTFIDF >= secondTFIDF) and (firstTFIDF <= thirdTFIDF) and (fourthTFIDF <= thirdTFIDF):
        output.at[documentId , 'Matched'] = 3
      elif  (fourthTFIDF >= secondTFIDF) and (firstTFIDF <= fourthTFIDF) and (fourthTFIDF >= thirdTFIDF):
        output.at[documentId , 'Matched'] = 4
      output.at[documentId , 'Lable'] = lables[output.at[documentId , 'Matched']-1]
    queryId += 1
  
  documentId += 1
