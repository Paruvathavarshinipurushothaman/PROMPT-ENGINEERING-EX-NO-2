# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## CODE
```
import pandas as pd df=pd.read_csv(r"C:\Users\Downloads\titanic_dataset.csv") df
df.shape
df.set_index("PassengerId",inplace=True) df
df.nunique
df['Sex'].value_counts()
df.Survived.unique()
df.rename(columns={"Sex":"Gender"},inplace=True)
import seaborn as sns
sns.countplot(data=df)
sns.countplot(x="Survived",hue="Gender",data=df)
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")
sns.boxplot(data=df)
df.boxplot(column="Survived",by="Gender")
sns.scatterplot(data=df)
sns.scatterplot(x=df['Age'],y=df['Fare'])
sns.jointplot(x='Age',y='Fare',data=df)
sns.jointplot(x='Age',y='Fare',data=df,kind="kde")
sns.jointplot(x='Age',y='Fare',data=df,kind="hist")
sns.catplot(x='Gender',col='Survived',data=df,kind='count',color='green')
sns.pairplot(data=df) image corr1=df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr1,annot=True)
sns.catplot(x='Gender',col='Survived',data=df,kind='count',hue="Pclass")
import matplotlib.pyplot as plt
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```

## OUTPUT


## RESULT
