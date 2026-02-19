# Ex-01_DS_Data_Cleansing
# REGISTER NO : 212224220054
# NAME : MADHAVAN K
# DATE : 19/02/2026

## AIM
To read the given data and perform data cleaning and save the cleaned data to a file. 

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. 
Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Get the information about the data
### STEP 3
Remove the null values from the data
### STEP 4
Save the Clean data to the file

# CODE and OUTPUT

```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
```
<img width="689" height="567" alt="Screenshot 2025-08-26 175401" src="https://github.com/user-attachments/assets/f5addb37-572c-43fd-ab5d-e6d8a259fd24" />

df.head()


<img width="670" height="230" alt="Screenshot 2025-08-26 175634" src="https://github.com/user-attachments/assets/8d92c6c2-3467-4970-9ac9-c14f380003f9" />

df.tail()


<img width="680" height="185" alt="Screenshot 2025-08-26 175800" src="https://github.com/user-attachments/assets/a943dcf1-fc0a-4573-9709-57e19682195d" />

df.isnull()


<img width="596" height="560" alt="Screenshot 2025-08-26 175946" src="https://github.com/user-attachments/assets/14b0851c-586a-4182-9e61-4eea73b8e2a2" />

df.isnull().sum()


<img width="571" height="378" alt="Screenshot 2025-08-26 180040" src="https://github.com/user-attachments/assets/766da06a-2953-44a1-b10e-2b3824ed2106" />

df.isnull().any()


<img width="619" height="387" alt="Screenshot 2025-08-26 180134" src="https://github.com/user-attachments/assets/e427db2c-fdd5-4bc3-b297-3ddf618407bf" />

df.dropna(axis=0)


<img width="716" height="378" alt="Screenshot 2025-08-26 180242" src="https://github.com/user-attachments/assets/353af0d8-fe59-48bc-91aa-a89424b53c5e" />

df.dropna(axis=1)


<img width="792" height="568" alt="Screenshot 2025-08-26 180338" src="https://github.com/user-attachments/assets/b2283856-a106-4509-993b-588cdb35cfdb" />

df.fillna(0)


<img width="765" height="568" alt="Screenshot 2025-08-26 180449" src="https://github.com/user-attachments/assets/ba277e93-f521-4948-b86c-3b22680b7122" />

df.fillna(method= 'ffill')


<img width="1057" height="598" alt="Screenshot 2025-08-26 180550" src="https://github.com/user-attachments/assets/b582e771-75fc-43a9-a9bd-f6e575f2e91d" />


df.fillna(method= 'bfill')


<img width="1046" height="593" alt="Screenshot 2025-08-26 180710" src="https://github.com/user-attachments/assets/9f8c3663-9d3a-4c8f-a5dc-70bd21443b9e" />

```
df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':908.00,'AVG':67.98})
```

<img width="834" height="562" alt="Screenshot 2025-08-26 180810" src="https://github.com/user-attachments/assets/14139c2f-c24e-4c89-adb1-b101ce4a133d" />

```
ir=pd.read_csv('iris.csv')
ir
```

<img width="575" height="362" alt="Screenshot 2025-08-26 180955" src="https://github.com/user-attachments/assets/ee47d7b8-32c7-46be-ab0e-5a2155ed3903" />


ir.describe()


<img width="528" height="263" alt="Screenshot 2025-08-26 181008" src="https://github.com/user-attachments/assets/74095793-0823-431d-8fe7-b7febd0774f5" />

```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```

<img width="580" height="396" alt="Screenshot 2025-08-26 181019" src="https://github.com/user-attachments/assets/52a054a7-c476-4032-80b4-fda409744ea8" />

```
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```

<img width="526" height="102" alt="Screenshot 2025-08-26 181033" src="https://github.com/user-attachments/assets/c4b2222e-05d6-4be9-95d3-551e6b68f019" />

```
rid=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid['sepal_width']
```

<img width="523" height="209" alt="Screenshot 2025-08-26 181044" src="https://github.com/user-attachments/assets/7999b74a-ba5d-4a39-8b12-f339fbfc5035" />

```
delid=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
delid
```

<img width="551" height="361" alt="Screenshot 2025-08-26 181058" src="https://github.com/user-attachments/assets/1d14eb4a-b18d-4da9-8ec8-da54853478c8" />


sns.boxplot(x='sepal_width',data=delid)


<img width="541" height="388" alt="Screenshot 2025-08-26 181108" src="https://github.com/user-attachments/assets/6f453c68-847f-4431-9ee0-22dfe901fd8d" />

```
import numpy as np
import scipy.stats as stats
z = np.abs(stats.zscore(ir['sepal_width']))
z
```

<img width="508" height="484" alt="Screenshot 2025-08-26 181120" src="https://github.com/user-attachments/assets/1136e5c9-c07c-43ee-a83f-0c7b42e54462" />

```
df1 = ir[z<3]
df1
```

<img width="529" height="374" alt="Screenshot 2025-08-26 181131" src="https://github.com/user-attachments/assets/96f05f58-3734-4f48-bbc4-5abb5360bb63" />

## RESULT:
 Thus the code executed successfully.
