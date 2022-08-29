
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#Ucitavanje fajla sa podacima popisanih gradjana
df = pd.read_csv('D:/Projekat/Prihodi gradjana 565-2016 Milan Djekic/Prihodi gradjana 565-2016/Prihodi gradjana/adult.csv')

#Mesto gde se cuvaju vizuelni prikazi iz projekta kada pokrenemo projekat
graph_folder_path = 'D:/Projekat/Prihodi gradjana 565-2016 Milan Djekic/Prihodi gradjana 565-2016/Prihodi gradjana/Grafici/'

#Analiza podataka sa nultim vrednostima
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')



#Konverzija podataka o prihodu u dve numericke vrednosti
df['income_level'] = np.where(df.income == '<=50K', 0, 1)

#Konverzija podataka o polu u numericke vrednosti
df['gender'] = df['sex'].map({'Male':0, 'Female':1}).astype(int)

#Konverzija podataka o rasi u numericke vrednosti i dodela vrednosti svakoj rasi
ethnicity_key = {'White':0, 'Black':1, 'Asian-Pac-Islander':2,
'Amer-Indian-Eskimo':3, 'Other':4}

df['ethnicity'] = df['race'].map(ethnicity_key).astype(int)

#Pretvaramo podatke u numericke vrednosti iz drzave odakle  poticu popisani gradjani
origin_key = {'?':0,'United-States':1, 'Mexico':2, 'Philippines':3,
'Germany':4, 'Canada':5, 'Puerto-Rico':6, 'El-Salvador':7, 
'India':8, 'Cuba':9, 'England':10,'Jamaica':11, 'South':12, 
'China':13, 'Italy':14, 'Dominican-Republic':15, 'Vietnam':16,
'Guatemala':17, 'Japan':18, 'Poland':19, 'Columbia':20, 'Taiwan':21,
'Haiti':22, 'Iran':23, 'Portugal':24, 'Nicaragua':25, 'Peru':26, 
'France':27, 'Greece':28, 'Ecuador':29, 'Ireland':30,'Hong':31,
'Trinadad&Tobago':32, 'Cambodia':33, 'Laos':34, 'Thailand':35, 
'Yugoslavia':36, 'Outlying-US(Guam-USVI-etc)':37, 'Hungary':38,
'Honduras':39, 'Scotland':40, 'Holand-Netherlands':41}

df['native_country'] = df['native.country'].map(origin_key).astype(int)

#Pretvaranje podataka o radnom odnosu i dodela vrednosti svakom

work_key = {'Private':0, 'Self-emp-not-inc':1, 'Local-gov':2, '?':3, 
'State-gov':4, 'Self-emp-inc':5, 'Federal-gov':6, 
'Without-pay':7,'Never-worked':8}

df['work'] = df['workclass'].map(work_key).astype(int)

#Pretvaramo podataka o bracnom statusu u numericke vrednosti
marital_status_key = {'Married-civ-spouse':0, 'Never-married':1, 'Divorced':2,
'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 
'Married-AF-spouse':6}

df['marital_status'] = df['marital.status'].map(marital_status_key).astype(int)

#Isto radimo i sa zanimanjima
occupation_key = {'Prof-specialty':0, 'Craft-repair':1, 'Exec-managerial':2, 
'Adm-clerical':3, 'Sales':4, 'Other-service':5,
'Machine-op-inspct':6, '?':7, 'Transport-moving':8, 
'Handlers-cleaners':9, 'Farming-fishing':10, 'Tech-support':11,
'Protective-serv':12, 'Priv-house-serv':13, 'Armed-Forces':14}

df['occupation'] = df['occupation'].map(occupation_key).astype(int)

#Takodje i sa porodicnim odnosima i funkcjama u istoj
relationship_key = {'Husband':0, 'Not-in-family':1, 'Own-child':2, 'Unmarried':3,
'Wife':4, 'Other-relative':5}

df['relationship'] = df['relationship'].map(relationship_key).astype(int)


df = df.drop(['income'], axis = 1)
df = df.drop(['sex'], axis = 1)
df = df.drop(['race'], axis = 1)
df = df.drop(['native.country'], axis = 1)
df = df.drop(['workclass'], axis = 1)
df = df.drop(['marital.status'], axis = 1)
df = df.drop(['education'], axis = 1)

#Radne sate delimo u tri kategorije i dodeljujemo im vrednosti: <40, 40, amd >40
df['hours.per.week'] = df['hours.per.week'].astype(int)
df.loc[df['hours.per.week'] < 40, 'hours.per.week'] = 0
df.loc[df['hours.per.week'] == 40, 'hours.per.week'] = 1
df.loc[df['hours.per.week'] > 40, 'hours.per.week'] = 2

#Sledi vizualizacija sredjenih podataka i provucenih kroz numericke vrednosti

#Stampamo tabelu ucestalosti stepena obrazovanja stanovnistva, gde selektira od 1-16 i prebrojava kroz podatke, i dalje stampa u programu
print(df['education.num'].value_counts())

#Izrada histograma starosti ispitanika
ageHist = plt.figure()
plt.title('Prihodi na osnovu godina')
df['age'].hist(bins = 20)
plt.savefig(graph_folder_path + 'prihodGodine.png')

#Izrada grafikona koji prikazuje prihod na osnovu pola ispitanika
incomeGenderBar = plt.figure()
incomeGenderBar = pd.crosstab(df['gender'], df['income_level'])
incomeGenderBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Prihodi na osnovu pola')
plt.savefig(graph_folder_path + 'prihodPol.png')

#Grafikon sa prikazom prihoda na osnovu obrazovanja
incomeEdBar = plt.figure()
incomeEdBar = pd.crosstab(df['education.num'], df['income_level'])
incomeEdBar.plot(kind = 'bar', color = ['red','green'], 
	grid = False, title = 'Prihodi na osnovu obrazovanja')
plt.savefig(graph_folder_path + 'prihodObrazovanje.png')

#Grafikon sa prikazom prihoda na osnovu zanimanja ili delatnosti
incomeOccGraph = plt.figure()
incomeOccGraph = pd.crosstab(df['occupation'], df['income_level'])
incomeOccGraph.plot(kind = 'bar', stacked = True, color = ['red', 'blue'],
	grid = False, title = 'Prihodi na osnovu zanimanja')
plt.savefig(graph_folder_path + 'prihodDelatnost.png')

#Grafikon sa prikazom prihoda na osnovu bracnog statusa
incomeMarriageBar = plt.figure()
incomeMarriageBar = pd.crosstab(df['marital_status'], df['income_level'])
incomeMarriageBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Prihodi na osnovu bracnog statusa')
plt.savefig(graph_folder_path + 'prihodBracniStatus.png')

#Grafikon sa prikazom prihoda na osnovu zemlje iz koje dolaze, domovine
incomeCountryBar = plt.figure()
incomeCountryBar = pd.crosstab(df['native_country'], df['income_level'])
incomeCountryBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Prihodi na osnovu zavicajne zemlje')
plt.savefig(graph_folder_path + 'prihodRodnaZemlja.png')

#Grafikon sa prikazom prihoda na osnuv radnih sati na nedeljnom nivou
incomeHoursBar = plt.figure()
incomeHoursBar = pd.crosstab(df['hours.per.week'], df['income_level'])
incomeHoursBar.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Prihodi na osnovu radnih sati u nedelji')
plt.savefig(graph_folder_path + 'prihodRadniSati.png')

#Pocinje model treninga
#Deli skup podataka na skup podataka za trening i testiranje
train, test = train_test_split(df, test_size = 0.3)

train_x = train.drop(['income_level'], axis = 1)
train_y = train['income_level']

test_x = test.drop(['income_level'], axis = 1)
test_y = test['income_level']

#******************************   Koristimo vise modela treninga kako bi dobili najbolji    ******************************

#Pocinjemo sa KNN algoritmom(algoritam k najblizih suseda), pa zato i koristimo biblioteku "from sklearn.neighbors import KNeighborsClassifier"
#KNN algoritam deluje tako sto pronalazi udaljenosti izmedju zadatog upita i svih primera u podacima birajuci navedene primere broja K koji su najblizi upitu a potom glasa za najbolju soluciju u odnosu na to da li je slucaj klasifikacije ili regresije.
k_values = np.arange(1, 25)
scores = []

for k in k_values:
	model = KNeighborsClassifier(n_neighbors = k)
	model.fit(train_x, train_y)
	KNN_prediction = model.predict(test_x)
	scores.append(metrics.accuracy_score(test_y, KNN_prediction))

print('Tacnost algoritma KNN:')
print(scores.index(max(scores)), max(scores))
print(' ')

#Algoritam stabla odlucivanja, pa zato koristimo biblioteku "from sklearn.tree import DecisionTreeClassifier"
#Stablo odlucivanja je alat za podrsku odluka koji koristi grafikon u obliku stabla ili model odluka i njihovih mogucih posledica, ukljucujuci i sansu ishoda dogadjaja, troskove resursa i korisnosti. To je jedan nacin da se prikaze algoritam.

model = DecisionTreeClassifier(class_weight = None, min_samples_leaf = 100, 
	random_state = 10)
model.fit(train_x, train_y)

DTC_prediction = model.predict(test_x)

print('Tacnost algoritma stabla odlucivanja:')
print(metrics.accuracy_score(test_y, DTC_prediction))
print(' ')

#Algoritam Xgboost, pa zato koristimo biblioteku from xgboost import XGBClassifie
#XG bust je algoritam koji se sve vise koristi prilikom obrade strukturnih i tabelarnih podataka, implementira stablo odluka sa povecanim gradijentima koji je dizajniran za bolju brzinu i performanse
XGBClassifier = XGBClassifier()
XGBClassifier.fit(train_x, train_y)

XGBC_prediction = XGBClassifier.predict(test_x)

print('Tacnost algoritma Xgboost:')
print(metrics.accuracy_score(test_y, XGBC_prediction))

#Algoritam catboost, pa koristimo biblioteku from catboost import CatBoostClassifier
#Algoritam catboost se koristi za rangiranje zadataka, predvidjanja i davanja preporuka, generalno radi na povecanju gradijenta na stablima odluka.
CBClassifier = CatBoostClassifier(learning_rate = 0.04,iterations=1000)
CBClassifier.fit(train_x, train_y)
CBC_prediction = CBClassifier.predict(test_x)

print(confusion_matrix(test_y,CBC_prediction)) #matrica konfuzije
print(classification_report(test_y,CBC_prediction))
print('Tacnost algoritma CatBoost: ')
print(metrics.accuracy_score(test_y, CBC_prediction)*100,'%')


#Algoritam MLPRegressor, pa koristimo biblioteku from sklearn.neural_network  import MLPClassifier

#MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
#alpha=0.0001, batch_size='auto', learning_rate='constant',
# learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
 #random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
# nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
 #beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
mlp = MLPClassifier(hidden_layer_sizes=(30,50),max_iter=1000 ,activation='tanh', solver = 'sgd', tol = -1)
mlp.fit(train_x, train_y)


predictions = mlp.predict(test_x)
print(confusion_matrix(test_y,predictions))
print(classification_report(test_y,predictions))
print('Tacnost algoritma MLP: ')
print(accuracy_score(test_y,predictions)*100,"%")






