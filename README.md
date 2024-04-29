## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
       import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![Screenshot 2024-04-29 144722](https://github.com/23013743/EXNO-3-DS/assets/161271714/1f851aa8-e5bb-4862-932f-7101e93257f8)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-04-29 144730](https://github.com/23013743/EXNO-3-DS/assets/161271714/52b823b6-39dc-49d6-879e-aa42440bb5be)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-04-29 144926](https://github.com/23013743/EXNO-3-DS/assets/161271714/6b257b52-8880-49b8-ba97-9b5b0ace511c)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-04-29 145057](https://github.com/23013743/EXNO-3-DS/assets/161271714/f98ef512-d43e-4066-8ee8-b8f4fcc8bfe2)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-04-29 145208](https://github.com/23013743/EXNO-3-DS/assets/161271714/7bfb77ed-7c12-402b-85b9-7abfb5c26ec5)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-04-29 145320](https://github.com/23013743/EXNO-3-DS/assets/161271714/ad1fc22e-a687-464c-b1fc-3916f9bd8027)
```
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2024-04-29 145602](https://github.com/23013743/EXNO-3-DS/assets/161271714/e3acd882-c225-4a90-baa7-deb4c061e508)

```
 from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```
![Screenshot 2024-04-29 145730](https://github.com/23013743/EXNO-3-DS/assets/161271714/055aa3b3-0a71-4914-9d01-e8b582e2746c)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![Screenshot 2024-04-29 145837](https://github.com/23013743/EXNO-3-DS/assets/161271714/a2aceae7-dbbf-49cd-b5b9-0ec8f87b0470)
```
df.skew()
```
![Screenshot 2024-04-29 145923](https://github.com/23013743/EXNO-3-DS/assets/161271714/a2450fbb-c2b4-4af7-a06e-74b7b5e54a32)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-04-29 150032](https://github.com/23013743/EXNO-3-DS/assets/161271714/0563347e-3c54-4bd8-b9d1-5040ba973619)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-04-29 150127](https://github.com/23013743/EXNO-3-DS/assets/161271714/dc61d780-0f50-4723-8e2f-e88c1fa1946f)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-04-29 150228](https://github.com/23013743/EXNO-3-DS/assets/161271714/e2fe7199-11a2-4cad-97ab-e8863d03e7d4)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-04-29 150337](https://github.com/23013743/EXNO-3-DS/assets/161271714/a0e417cc-9da3-43bd-98e3-d63375d74a04)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-04-29 150425](https://github.com/23013743/EXNO-3-DS/assets/161271714/fddf804a-979b-4d99-b3cf-3cce929a1c19)
```
df.skew()
```
![Screenshot 2024-04-29 150604](https://github.com/23013743/EXNO-3-DS/assets/161271714/59ad0d2a-0ffb-4e2c-bc8e-dc3cf862fa44)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2024-04-29 150642](https://github.com/23013743/EXNO-3-DS/assets/161271714/3b1f9072-8b56-46d5-b788-165b0a9d8593)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```
![Screenshot 2024-04-29 150853](https://github.com/23013743/EXNO-3-DS/assets/161271714/c94f34c3-259c-46c3-8425-d6d85ad78b40)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
![Screenshot 2024-04-29 150940](https://github.com/23013743/EXNO-3-DS/assets/161271714/b396d2aa-ec1e-4ed4-84fd-8abce467668b)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-04-29 151110](https://github.com/23013743/EXNO-3-DS/assets/161271714/ded79d2c-4ca3-4d6f-bd64-bd4094009d9b)
````
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

````
![Screenshot 2024-04-29 151157](https://github.com/23013743/EXNO-3-DS/assets/161271714/54c5c95a-d83c-477f-b4c2-5bc94101c44b)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```
![Screenshot 2024-04-29 151308](https://github.com/23013743/EXNO-3-DS/assets/161271714/a76f632b-8804-4e9b-b37c-037237e413b3)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![Screenshot 2024-04-29 151428](https://github.com/23013743/EXNO-3-DS/assets/161271714/96bb590e-a5d1-4501-82fd-4654c8ddca5f)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![Screenshot 2024-04-29 151512](https://github.com/23013743/EXNO-3-DS/assets/161271714/9b522777-ac34-4941-a58e-59e88c13a2b6)

# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
