from sklearn.tree import export_text
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

r = export_text(clf, feature_names=feature_names)

df  = pd.read_csv('dt.csv')
#df  = df.dropna(axis = 1,  inplace=False)

target = df['TYPE']
df.drop( 'TYPE' , axis='columns', inplace = True)

df2 = df
df2 = df.select_dtypes(include=['object'])

from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
d = defaultdict(LabelEncoder)


# Encoding the variable
fit = df2.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
df2 = df2.apply(lambda x: d[x.name].transform(x))

df[list(df2.columns)] = df2[list(df2.columns)] 

df  = df.fillna(df.mode().iloc[0])



X = df
y = target

feature_names = list(X.columns)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
from sklearn.tree import export_text



# Rules

r = export_text(clf, feature_names=feature_names)
f = open("Rules_set.txt", "w")
f.write(r)

file1 = open("Rules_set.txt","r")
data = file1.readlines()
    
dic = {}
first = None

for line in data:
    if( 'class' in line):
        #print(line.index('class'))
        rule = ' and '.join(list(dic.values()))
        rule = rule + ' ' + line[line.index('class'):]
        print(rule.strip())
        
    else:
        for char in line:
            if char.isalpha():
                index = line.index(char)
                if first == None:
                    first = index
                if first == index:
                    dic = {}
                dic[index] = f'({line[index:].strip()})'
                break
