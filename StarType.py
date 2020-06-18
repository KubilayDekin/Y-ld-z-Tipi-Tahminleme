#Brown Dwarf -> Star Type = 0
#Red Dwarf -> Star Type = 1
#White Dwarf-> Star Type = 2
#Main Sequence -> Star Type = 3
#Supergiant -> Star Type = 4
#Hypergiant -> Star Type = 5 
# Python version
print('Versiyonlar')
print('_____________________________ \n')
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
print('_____________________________ \n')

# compare algorithms
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#veri setini yüklüyoruz
url = "6 class csv.csv"
names = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type','Star color','Spectral Class' ]
dataset = read_csv(url,names=names)

#veri setini uygun hale getiriyoruz
#Star color sütunu string tipinde olduğu için değiştiriyoruz
color=LabelEncoder()
dataset['Star color']=color.fit_transform(dataset['Star color'])
#Spectral Class sütunu string tipinde olduğu için değiştiriyoruz
spec_type=LabelEncoder()
dataset['Spectral Class']=spec_type.fit_transform(dataset['Spectral Class'])

#Star type bulmaya çalışıyoruz
array = dataset.values
X = array[:,0:7]
y = array[:,4]

#eğitim ve doğrulama
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)
# algoritmalar
print('Uygun Algoritma Denemesi')
print('_____________________________ \n')
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#en iyi modeli seçmek için her modelin başarı oranlarını buluyoruz 
results = []
names = []

for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# algoritmaları karşılaştırıyoruz
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
print('_____________________________ \n')
print('En iyi Algoritma= CART \n\n')
#tahminleme işlemi
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('accuracy_score basladi \n')
print('_____________________________ \n')
print('Accuracy Score=',accuracy_score(Y_validation, predictions))
print('_____________________________ \n')
print('Confusion Matrix \n',confusion_matrix(Y_validation, predictions))
print('_____________________________ \n')
print(dataset.groupby('Star type').size())
print(classification_report(Y_validation, predictions))