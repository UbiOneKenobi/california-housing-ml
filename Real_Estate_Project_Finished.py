from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

# Caricamento del dataset da internet
# Questa funzione scarica il dataset solo se non è già presente in locale,
# poi estrae il file .tgz e restituisce il contenuto del CSV come DataFrame pandas.
def load_housing_data():
    # Percorso locale in cui salviamo l'archivio compresso del dataset.
    tarball_path = Path("datasets/housing.tgz")

    # Se l'archivio non esiste ancora, creiamo la cartella e scarichiamo il file.
    # In questo modo non riscarichiamo il dataset a ogni esecuzione del programma.
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

    # Apriamo l'archivio .tgz ed estraiamo i file nella cartella datasets.
    # Dopo l'estrazione sarà disponibile il file datasets/housing/housing.csv.
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")

    # Leggiamo il CSV con pandas e restituiamo un DataFrame.
    # Da questo momento possiamo usare i metodi di pandas, ad esempio:
    # housing.info(), housing.describe(), housing.head(), ecc.
    return pd.read_csv(Path("datasets/housing/housing.csv"))


# Esegue la funzione e salva il DataFrame nella variabile housing.
housing = load_housing_data()


# Informazioni generali sul dataset
"""
print(housing.head())
# Mostra le prime 5 righe del dataset.

print(housing.info())
# Mostra un riepilogo del DataFrame: numero di righe, colonne, valori non nulli
# e tipo di dato di ogni colonna.

print(housing["ocean_proximity"].value_counts())
# Mostra quante righe appartengono a ciascuna categoria della colonna ocean_proximity.

print(housing.describe())
# Mostra statistiche descrittive sulle colonne numeriche: media, deviazione standard,
# minimo, massimo e quartili.
"""


import matplotlib.pyplot as plt

# Impostazioni grafiche usate per rendere più leggibili i grafici.
plt.rc('font', size=8)
plt.rc('axes', labelsize=8, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=10)


# Istogrammi delle colonne numeriche
"""
housing.hist(bins=50, figsize=(12, 8))
plt.show()
"""
# housing.hist() crea un istogramma per ciascuna colonna numerica.
# bins=50 significa che i valori di ogni colonna vengono divisi in 50 intervalli.
# Questo aiuta a capire la distribuzione dei dati: valori concentrati, outlier,
# scale diverse tra colonne e possibili limiti artificiali nei dati.


import numpy as np

# Creazione manuale di training set e test set usando indici casuali
"""
def shuffle_and_split_data(data, test_ratio):
    # data: DataFrame da dividere.
    # test_ratio: percentuale del dataset da mettere nel test set.
    # Esempio: test_ratio=0.2 significa 20% test set e 80% training set.

    # Impostiamo un seed per rendere riproducibile la divisione.
    # Con lo stesso dataset nello stesso ordine, otterremo sempre lo stesso split.
    np.random.seed(42)

    # Creiamo un array contenente tutti gli indici di riga e li mischiamo casualmente.
    shuffled_indices = np.random.permutation(len(data))

    # Calcoliamo quante righe devono andare nel test set.
    test_set_size = int(len(data) * test_ratio)

    # I primi indici finiscono nel test set, gli altri nel training set.
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # iloc seleziona le righe in base alla loro posizione numerica.
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = shuffle_and_split_data(housing, 0.2)

# Limite di questo metodo:
# anche se il seed rende lo split riproducibile sullo stesso dataset,
# lo split può cambiare se il dataset viene aggiornato, riordinato o se vengono aggiunte/rimosse righe.
# Per avere uno split più stabile nel tempo si può assegnare ogni riga al train/test set
# in base a un identificatore stabile della riga.
"""


# Creazione train/test set usando un hash CRC32
"""
from zlib import crc32


def is_id_in_test_set(identifier, test_ratio):
    # Questa funzione restituisce True se la riga deve andare nel test set.
    # L'idea è usare un identificatore stabile della riga, non la posizione casuale.

    # crc32 calcola un valore hash deterministico: dato lo stesso identifier,
    # restituisce sempre lo stesso numero nel range 0 ... 2^32 - 1.
    # Non è un numero davvero casuale, ma si comporta come una distribuzione pseudo-casuale.

    # Se il valore hash cade nella prima porzione del range, la riga va nel test set.
    # Con test_ratio=0.2, circa il 20% degli identificatori finirà nel test set.
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    # id_column è la colonna che contiene un identificatore stabile per ogni riga.
    ids = data[id_column]

    # apply applica la funzione a ogni valore della Series ids.
    # lambda id_: ... crea una piccola funzione anonima che prende un id
    # e restituisce True se quell'id deve stare nel test set.
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    # data.loc[in_test_set] seleziona le righe da mettere nel test set.
    # data.loc[~in_test_set] seleziona il complemento, cioè le righe del training set.
    return data.loc[~in_test_set], data.loc[in_test_set]


# Il dataset non contiene una colonna ID vera e propria.
# Qui creiamo una colonna index a partire dall'indice delle righe.
# Attenzione: questo funziona bene solo se le nuove righe vengono aggiunte in fondo
# e l'ordine delle righe già esistenti non cambia.
housing_with_id = housing.reset_index()

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
"""


# Creazione train/test set con la funzione pronta di scikit-learn
"""
from sklearn.model_selection import train_test_split

# train_test_split divide automaticamente il dataset in training set e test set.
# random_state=42 rende lo split riproducibile se il dataset resta uguale.
# Anche qui, però, se il dataset cambia o viene riordinato, alcune righe possono finire
# in un set diverso rispetto alle esecuzioni precedenti.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
"""


# Creazione della colonna income_cat
# median_income è una variabile molto importante per predire median_house_value.
# Per questo creiamo categorie di reddito e le usiamo per fare uno split stratificato:
# vogliamo che training set e test set abbiano proporzioni simili di distretti
# a basso, medio e alto reddito.
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Visualizzazione della distribuzione delle categorie di reddito
"""
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
"""
# Il grafico mostra quante righe appartengono a ogni fascia di reddito.
# Serve a controllare se alcune categorie sono molto più frequenti di altre.


from sklearn.model_selection import StratifiedShuffleSplit

# Split stratificato con StratifiedShuffleSplit
"""
# StratifiedShuffleSplit crea più coppie train/test.
# n_splits=10 significa: genera 10 possibili divisioni diverse.
# test_size=0.2 significa: ogni test set contiene il 20% delle righe.
# stratify, in questo caso, è dato da housing["income_cat"].
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# Usiamo la prima delle 10 divisioni generate.
strat_train_set, strat_test_set = strat_splits[0]
"""


from sklearn.model_selection import train_test_split

# Modo più compatto per ottenere direttamente uno split stratificato.
# stratify=housing["income_cat"] dice a scikit-learn di mantenere nel train e nel test
# circa le stesse proporzioni delle categorie di reddito presenti nel dataset completo.
strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42,
    stratify=housing["income_cat"]
)

# Verifica delle proporzioni di income_cat nel dataset completo, nel train set e nel test set
"""
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(housing["income_cat"].value_counts() / len(housing))
"""
# Se la stratificazione è corretta, le percentuali delle categorie saranno molto simili
# nei tre output.


# Rimuoviamo income_cat perché era una colonna ausiliaria usata solo per lo split.
# Non deve rimanere tra le feature del modello, altrimenti aggiungeremmo una variabile artificiale.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Da qui in poi lavoriamo su una copia del training set.
# È importante esplorare e trasformare solo il training set, non il test set,
# perché il test set deve restare separato fino alla valutazione finale del modello.
housing = strat_train_set.copy()


# Visualizzazione geografica semplice
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
# plt.show()

# Con alpha=0.2 i punti sono semitrasparenti: le aree con molti punti sovrapposti
# appaiono più dense e quindi più facili da individuare.
# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()


# Visualizzazione geografica con popolazione e valore medio delle case
"""
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    grid=True,
    s=housing["population"] / 100,   # dimensione del punto proporzionale alla popolazione
    label="population",
    c="median_house_value",          # colore del punto proporzionale al valore medio delle case
    cmap="plasma",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7)
)
plt.show()
"""
# Questo grafico permette di vedere contemporaneamente:
# - posizione geografica dei distretti;
# - densità/popolazione tramite la dimensione dei punti;
# - valore medio delle case tramite il colore.


# Correlazioni lineari tra le variabili numeriche
# corr() calcola il coefficiente di correlazione di Pearson tra coppie di colonne numeriche.
# Valori vicini a 1 indicano correlazione positiva forte;
# valori vicini a -1 indicano correlazione negativa forte;
# valori vicini a 0 indicano assenza di relazione lineare evidente.
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)


from pandas.plotting import scatter_matrix

# Scatter matrix per osservare visivamente le relazioni tra alcune variabili importanti.
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))


# median_income sembra avere la relazione più evidente con median_house_value,
# quindi lo analizziamo con uno scatter plot più dettagliato.
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)

# Feature engineering: creazione di variabili più informative
# Alcune colonne grezze sono poco significative da sole.
# Per esempio, total_rooms dipende molto dalla dimensione del distretto;
# rooms_per_house è spesso più utile perché descrive la dimensione media delle abitazioni.
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["people_per_house"] = housing["population"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]

# Ricalcoliamo le correlazioni dopo aver aggiunto le nuove feature.
# Se una nuova feature ha una correlazione più alta in valore assoluto con median_house_value,
# potrebbe essere utile al modello.
# Attenzione: la correlazione misura solo relazioni lineari e non dimostra causalità.
corr_matrix = housing.corr(numeric_only=True)
#print(corr_matrix["median_house_value"].sort_values(ascending=False))


# questo serve per separare i label dal dataset principale
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# print(housing.info())
# print(housing_labels.info())


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy= "median")

# andiamo a creare un nuovo database che contenga unicamente categorie di valori numerici
housing_num = housing.select_dtypes(include = [np.number])

# il metodo fit dell'imputer calcola le mediane di housing_num
imputer.fit(housing_num)

# con imputer.transform i valori mancanti del dataset vengono sostituiti con le mediane
X = imputer.transform(housing_num)

# dato che imputer.transform restituisce un array Numpy, allora lo trasformiamo in un database Panda
housing_tr = pd.DataFrame(X , columns = housing_num.columns , index  = housing_num.index)


housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

# usando l'OneHotEncoder andiamo a creare un nuovo database per ogni diverso valore della categoria ocean_proximity 
# e se l'indice lo presenta come valore, allora mostra 1, al contrario 0
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# dato che i range dei vari parametri variano molto per intensità 
# ( il numero totale di stanze va da 6 a 39.320 mentre il median_income va da 0 a 15)
# perciò normalizziamo tutti i valori di tutti i parametri da 0 a 1 per far lavorare meglio l'algoritmo

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range= (-1 , 1))

housing_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# un altro metodo per riportare i valori in un range accettabile è utilizzando la standardizzazione
# come si può notare non c'è un range di un valori preimpostato ( come -1 e 1 ne MinMaxScaler)
# questo algoritmo è però meno condizionato da valori estremi particolarmente piccoli o grandi

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

housing_num_std_scaled = std_scaler.fit_transform(housing_num)



# questa è una funzione custom che permette di svolgere il log per dei dati e impostarne l'exp come inversa

from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log , inverse_func= np.exp)

log_pop = log_transformer.transform(housing["population"])


# importiamo rbf_kernel prima di utilizzarlo all'interno del FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel

# creiamo un transformer che calcola la similarità fra housing_median_age e il valore 35
# gamma determina quanto velocemente la similarità diminuisce allontanandosi da 35
rbf_transformer = FunctionTransformer(rbf_kernel,
kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])


# possiamo ottenere lo stesso risultato anche utilizzando direttamente rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# questa classe ricrea in modo semplificato il funzionamento dello StandardScaler
# fit calcola media e deviazione standard, mentre transform standardizza i dati
class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean
    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!
    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    

from sklearn.cluster import KMeans

# questa classe trova i centri dei cluster geografici con KMeans e poi trasforma
# latitudine e longitudine nella similarità RBF rispetto a ciascun centro del cluster
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# creiamo 10 cluster usando latitudine e longitudine
# sample_weight assegna un peso maggiore alle zone con median_house_value più elevato
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight=housing_labels)

    
# quando bisogna fare numerose trasformazioni sui dati allora conviene usare una Pipeline che applica le varie 
# operazioni ai dati in ordine
# in questo caso i dati mancanti verranno inizialmente riempiti dalla mediana, poi verrano standardizzati
# possiamo anche aggiungere log_trasformer nel caso volessi avere il logaritmo dei dati

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy= "median" ), StandardScaler())


# per non dover lavorare con le singole colonne in base al loro tipo, andiamo a utilizzare ColumnTransformer che, date le colonne
# che hanno attributi numerici e attributi categorici, applica a loro le varie pipeline
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms","total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(SimpleImputer(strategy= "most_frequent"), OneHotEncoder(handle_unknown= "ignore"))

# ogni tupla indica il nome della trasformazione, la pipeline da applicare e le colonne interessate
preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", cat_pipeline , cat_attribs)])


# per semplificare ancora il processo, invece di dover elencare le colonne manualmente, possiamo usare dtype che seleziona in automatico
# le colonne in baso al loro tipo di dato

from sklearn.compose import make_column_selector , make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include = np.number)),
    (cat_pipeline, make_column_selector(dtype_include = object))
)

# la preparazione dei dati è finita, ora possiamo applicarne la trasformazione

housing_prepared = preprocessing.fit_transform(housing)



#pipeline finale

# questa funzione calcola il rapporto fra le due colonne ricevute dalla pipeline
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# assegniamo il nome ratio alla nuova colonna creata dal FunctionTransformer
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# prima riempiamo i valori mancanti, poi calcoliamo il rapporto e infine lo standardizziamo
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

# applichiamo il logaritmo alle colonne con distribuzione a coda lunga e poi le standardizziamo
# one-to-one indica che ogni colonna in entrata produce una colonna in uscita con lo stesso nome
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# trasforma latitudine e longitudine nella similarità rispetto ai 10 cluster geografici
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# questa pipeline viene applicata alle colonne numeriche che non hanno una trasformazione specifica
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

# applichiamo trasformazioni diverse a gruppi diversi di colonne
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age 


# FASE DI TRAINING
# usiamo inizialmente il modello di Regressione Lineare 
from sklearn.linear_model import LinearRegression

# creaiamo una pipeline in cui i dati dovranni prima essere processati e poi verrano usati nel modello di training
lin_reg = make_pipeline(preprocessing, LinearRegression())

# applichiamo il modello ad housing dandogli come labels quello di housing 
lin_reg.fit(housing , housing_labels)

housing_predictions = lin_reg.predict(housing)

# mostriamo i primi 5 risultati stimati ed i veri 5 risultati
#   print( housing_predictions[:5].round(-2) )
#   print( housing_labels[:5].round(-2))

# usando l'RMSE notiamo come ci sia un errore medio di 68.000$, un errore molto elevato dato che il modello è fin troppo semplice
# per i dati forniti
from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
#   print(lin_rmse)

# dato che la Regressione Lineare è troppo semplice cambiamo modello nel DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing , DecisionTreeRegressor(random_state= 42))
tree_reg.fit(housing , housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)

#   print(tree_rmse)
# un risultato 0 non significa che abbiamo 0 errori, ma che abbiamo probabilmente scelto un modello errato, forse troppo potente


# per valutare meglio il nostro modello ha però senso fare più test nel trainin set
# la cross validation permette di dividere il training set in 10 parti e allenare il modello con le 9 parti e di verificarlo con la 
# parte rimanente, ripetendo questo processo 10 volte

from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring = "neg_root_mean_squared_error", cv = 10)

#print(pd.Series(tree_rmses).describe())

# come possiamo notare adesso l'errore è presente, questo è dato dal fatot che prima abbiamo allenato il decision_tree su tutto housing
# e successivamente l'abbiamo testato su tutto housing, quindi riusciva a ricorda a memoria i pattern.
# in questo caso il modello viene allenato su 9/10 di housing e testato su 1/10 così da non poter andare a memoria 


# proviamo un altro modello, il modello foresta che prende numerosi tree regressor e li traina su piccole parti di housing

from sklearn.ensemble import RandomForestRegressor

# forest_reg = make_pipeline(preprocessing , RandomForestRegressor(random_state = 42, n_jobs = 1))

# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring ="neg_root_mean_squared_error", cv = 3, verbose = 2 )

#   print(pd.Series(forest_rmses).describe())


# il risultato è migliore rispetto agli altri modello
# il modello comunque overfitta i dati, perciò rimane la possibilità di lavorare sugli hyperparametri


# piuttosto di andare a modificare gli iperparametri manualmente, usiamo la funzione GridSearchCV che permette di fare diversi test
# in modo rapido

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
# notiamo che gli iperparametri modificati sono n_clusters del preprocessing e max_features del modello forest

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error', verbose = 2)
#   grid_search.fit(housing, housing_labels)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# RANDOMIZED SEARCH
# Cerca una buona combinazione di iperparametri estraendo valori casuali
# dagli intervalli indicati. A differenza della Grid Search, non prova tutte
# le combinazioni: il numero di tentativi è stabilito tramite n_iter.

# Intervalli da cui la Randomized Search estrarrà casualmente gli iperparametri.
# randint include "low" ma esclude "high": n_clusters sarà 3-49,
# mentre max_features sarà 2-19.
param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20)
}


# Prova 10 combinazioni casuali degli iperparametri.
# Ogni combinazione viene valutata con una cross-validation a 3 fold.
rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,                              # Numero di combinazioni casuali provate
    cv=3,                                   # Ogni combinazione viene allenata e valutata 3 volte
    scoring="neg_root_mean_squared_error",  # Valuta il modello usando l'opposto dell'RMSE
    random_state=42                         # Rende ripetibili le estrazioni casuali
)

# Avvia la ricerca: in totale 10 × 3 = 30 allenamenti.
rnd_search.fit(housing, housing_labels)

# questo serve per trovare gli argomenti più importanti
final_model = rnd_search.best_estimator_  # includes preprocessing


# andiamo a svolgere il test con il test set, dividendo i labels
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)

#print(final_rmse)

import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")