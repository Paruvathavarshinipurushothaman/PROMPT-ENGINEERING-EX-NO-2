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
<img width="1057" height="401" alt="image" src="https://github.com/user-attachments/assets/8624dc0f-912e-4c61-80d4-432f1025226f" />
<img width="92" height="32" alt="image" src="https://github.com/user-attachments/assets/731b5f60-ea69-4759-9dfb-d1889e8fc698" />
<img width="1022" height="412" alt="image" src="https://github.com/user-attachments/assets/fdbf720d-8c47-4a65-ba7d-33b0c054c1e8" />
<img width="636" height="501" alt="image" src="https://github.com/user-attachments/assets/15f51413-c031-4f2f-8ee5-b083a1f50c7d" />
<img width="557" height="287" alt="image" src="https://github.com/user-attachments/assets/dce0010e-cdbe-4a8d-b268-570ab6486557" />
<img width="246" height="82" alt="image" src="https://github.com/user-attachments/assets/0c9f44c5-197a-4460-8852-03a2e319b5f3" />
<img width="138" height="27" alt="image" src="https://github.com/user-attachments/assets/bf3aeab4-b193-4d0f-954b-1f8697898e23" />
<img width="1057" height="431" alt="image" src="https://github.com/user-attachments/assets/fe68012e-7558-4b51-9c70-b37cefea1fcc" />
<img width="707" height="527" alt="image" src="https://github.com/user-attachments/assets/98c75bef-a512-4333-b8e2-dc45335e012c" />
<img width="728" height="536" alt="image" src="https://github.com/user-attachments/assets/43cc0737-9356-42c1-917f-c83653ee0182" />
<img width="718" height="563" alt="image" src="https://github.com/user-attachments/assets/8de1479e-bd61-404b-9849-ac4b19cc22d3" />
<img width="696" height="562" alt="image" src="https://github.com/user-attachments/assets/430aa5a3-592c-4959-a3b8-ba473d2e8a35" />
<img width="616" height="452" alt="image" src="https://github.com/user-attachments/assets/d40651c4-07e9-4fc1-94f1-bc1c45af758a" />
<img width="632" height="462" alt="image" src="https://github.com/user-attachments/assets/448765ed-72b3-4c45-9e96-36ce02398655" />
<img width="800" height="572" alt="image" src="https://github.com/user-attachments/assets/40a3b02a-c054-4edd-8320-7bdbb02738dd" />
<img width="590" height="446" alt="image" src="https://github.com/user-attachments/assets/8afacf0b-068a-47e3-a93c-f400129f6f6d" />
<img width="666" height="607" alt="image" src="https://github.com/user-attachments/assets/7dea9d13-aaa7-474f-96b2-302ccbddeabd" />
<img width="628" height="590" alt="image" src="https://github.com/user-attachments/assets/7f95a977-dfbc-479a-ba9b-a8544dc8c03b" />
<img width="785" height="738" alt="image" src="https://github.com/user-attachments/assets/0db49ccd-e9bd-4b94-b1ed-28f12ecacb5f" />
<img width="1027" height="496" alt="image" src="https://github.com/user-attachments/assets/2433c173-5669-4a2d-a10b-840b9afd2be2" />
<img width="1081" height="593" alt="image" src="https://github.com/user-attachments/assets/3ab66894-1855-489e-b624-065800e3af6b" />
<img width="1066" height="472" alt="image" src="https://github.com/user-attachments/assets/b05db47b-3d74-45f3-9718-73d663054f8c" />
<img width="807" height="605" alt="image" src="https://github.com/user-attachments/assets/77693137-8f57-42cc-82db-b6f6cc450ff8" />
<img width="1031" height="471" alt="image" src="https://github.com/user-attachments/assets/9ae33a28-8af3-4405-bb89-c836090c9414" />
<img width="865" height="521" alt="image" src="https://github.com/user-attachments/assets/29fb513e-9974-425f-b945-621e88d948e8" />

## RESULT
Thus, performing Exploratory Data Analysis on the given data set.

