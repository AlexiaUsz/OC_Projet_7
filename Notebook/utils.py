import pickle
import pandas as pd
import gc
import numpy as np

def clean_feature_names(df):
    """
    Nettoie les noms de colonnes du DataFrame.

    Cette fonction supprime les caractères spéciaux des noms de colonnes d'un DataFrame
    et les remplace par des caractères alphanumériques ou des underscores (_).
    
    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame dont les noms de colonnes doivent être nettoyés.
    
    Retours
    -------
    df : pandas.DataFrame
        DataFrame avec les noms de colonnes nettoyés.
    
    Exemple
    -------
    >>> df = clean_feature_names(df)
    """
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
    return df

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    """
    Encodage One-Hot des colonnes catégoriques avec get_dummies.
    
    Cette fonction effectue un encodage One-Hot sur les colonnes catégoriques d'un DataFrame en utilisant `pandas.get_dummies`.
    Les colonnes catégoriques sont détectées en recherchant les colonnes de type `object`.
    
    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à encoder.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage.
        Par défaut, True.
    
    Retours
    -------
    df : pandas.DataFrame
        DataFrame avec les colonnes catégoriques encodées en One-Hot.
    new_columns : list of str
        Liste des nouvelles colonnes créées après l'encodage One-Hot.
        
    Exemple
    -------
    >>> df, new_cols = one_hot_encoder(df, nan_as_category=False)
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(app_data, nan_as_category = False):
    """
    Prétraitement des fichiers application_train.csv et application_test.csv.
    
    Cette fonction lit les données des fichiers CSV `application_train.csv` et `application_test.csv`,
    les fusionne en un seul DataFrame, supprime les applications avec un code de genre 'XNA' (dans l'ensemble d'entraînement),
    effectue des encodages binaire et One-Hot sur certaines fonctionnalités catégoriques, et crée de nouvelles fonctionnalités basées sur les données existantes.
    
    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire des fichiers CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
    
    Retours
    -------
    df : pandas.DataFrame
        DataFrame fusionné contenant les données prétraitées de `application_train.csv` et `application_test.csv`.
        
    Exemple
    -------
    >>> df = application_train_test(num_rows=1000, nan_as_category=True)
    """
    # Read data and merge
    #df = df_application_train
    #test_df = df_application_test
    #print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    #df = pd.concat([df, test_df], ignore_index=True)
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    app_data = app_data[app_data['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        #app_data[bin_feature], uniques = pd.factorize(app_data[bin_feature])
        app_data.loc[:, bin_feature], uniques = pd.factorize(app_data[bin_feature])
    # Categorical features with One-Hot encode
    app_data, cat_cols = one_hot_encoder(app_data, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    app_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    app_data['DAYS_EMPLOYED_PERC'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['INCOME_CREDIT_PERC'] = app_data['AMT_INCOME_TOTAL'] / app_data['AMT_CREDIT']
    app_data['INCOME_PER_PERSON'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    app_data['ANNUITY_INCOME_PERC'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    app_data['PAYMENT_RATE'] = app_data['AMT_ANNUITY'] / app_data['AMT_CREDIT']
    return app_data

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(df_bureau, df_bureau_balance, num_rows = None, nan_as_category = True):
    """
    Prétraitement des fichiers bureau.csv et bureau_balance.csv.
    
    Cette fonction lit les données des fichiers CSV `bureau.csv` et `bureau_balance.csv`, 
    effectue des encodages One-Hot sur les colonnes catégoriques, réalise des agrégations 
    sur les données de `bureau_balance.csv`, et fusionne les résultats avec les données 
    de `bureau.csv`. Elle effectue également des agrégations sur les caractéristiques 
    numériques et catégoriques des ensembles de données combinés, et crée des agrégations 
    distinctes pour les crédits actifs et fermés.

    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire des fichiers CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
        Par défaut, True.
    
    Retours
    -------
    bureau_agg : pandas.DataFrame
        DataFrame contenant les agrégations de `bureau.csv` et `bureau_balance.csv` 
        et les caractéristiques associées aux crédits actifs et fermés.
        
    Exemple
    -------
    >>> bureau_agg = bureau_and_balance(num_rows=1000, nan_as_category=True)
    """
    #bureau = pd.read_csv('../data/bureau.csv', nrows = num_rows)
    bureau = df_bureau
    bb = df_bureau_balance
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    #del bb, bb_agg
    #gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat: 
        cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    #del closed, closed_agg, bureau
    #gc.collect()
    bureau_data = bureau_agg
    return bureau_data

# Preprocess previous_applications.csv
def previous_applications(df_previous_application, nan_as_category = True):
    """
    Prétraitement des fichiers previous_applications.csv.

    Cette fonction lit les données du fichier CSV `previous_application.csv`, effectue
    des encodages One-Hot sur les colonnes catégoriques, remplace certaines valeurs spécifiques
    de `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, 
    et `DAYS_TERMINATION` par NaN, ajoute une fonctionnalité calculée à partir du ratio entre
    `AMT_APPLICATION` et `AMT_CREDIT`, et réalise des agrégations sur les données des
    applications précédentes. Elle crée également des agrégations distinctes pour les applications
    approuvées et refusées.

    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire du fichier CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
        Par défaut, True.

    Retours
    -------
    prev_agg : pandas.DataFrame
        DataFrame contenant les agrégations des applications précédentes, ainsi que
        les caractéristiques associées aux applications approuvées et refusées.

    Exemple
    -------
    >>> prev_agg = previous_applications(num_rows=1000, nan_as_category=True)
    """
    prev = df_previous_application
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= nan_as_category)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    prev_data = prev_agg
    return prev_data

# Preprocess POS_CASH_balance.csv
def pos_cash(df_POS_CASH_balance, num_rows = None, nan_as_category = True):
    """
    Prétraitement du fichier POS_CASH_balance.csv.

    Cette fonction lit les données du fichier CSV `POS_CASH_balance.csv`, effectue
    des encodages One-Hot sur les colonnes catégoriques, et réalise des agrégations
    sur les données du solde POS_CASH. Elle calcule également le nombre de comptes
    POS_CASH par client.

    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire du fichier CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
        Par défaut, True.

    Retours
    -------
    pos_agg : pandas.DataFrame
        DataFrame contenant les agrégations des données POS_CASH et le nombre
        de comptes POS_CASH par client.

    Exemple
    -------
    >>> pos_agg = pos_cash(num_rows=1000, nan_as_category=True)
    """
    pos = df_POS_CASH_balance
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    pos_data = pos_agg
    return pos_data

# Preprocess installments_payments.csv
def installments_payments(df_installments_payments, num_rows = None, nan_as_category = True):
    """
    Prétraitement du fichier installments_payments.csv.

    Cette fonction lit les données du fichier CSV `installments_payments.csv`, effectue
    des encodages One-Hot sur les colonnes catégoriques, calcule le pourcentage et la différence
    de paiement pour chaque versement, et calcule le nombre de jours de retard et d'avance de paiement.
    Elle réalise également des agrégations sur les données des paiements d'installments.

    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire du fichier CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
        Par défaut, True.

    Retours
    -------
    ins_agg : pandas.DataFrame
        DataFrame contenant les agrégations des paiements d'installments et le nombre
        de comptes d'installments par client.

    Exemple
    -------
    >>> ins_agg = installments_payments(num_rows=1000, nan_as_category=True)
    """
    ins = df_installments_payments
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    ins_data = ins_agg
    return ins_data

# Preprocess credit_card_balance.csv
def credit_card_balance(df_credit_card_balance, num_rows = None, nan_as_category = True):
    """
    Prétraitement du fichier credit_card_balance.csv.

    Cette fonction lit les données du fichier CSV `credit_card_balance.csv`, effectue
    des encodages One-Hot sur les colonnes catégoriques, et réalise des agrégations
    sur les données du solde de carte de crédit. Elle calcule également le nombre de lignes
    de carte de crédit par client et remplace les valeurs manquantes par la moyenne
    des colonnes numériques.

    Paramètres
    ----------
    num_rows : int, optionnel
        Nombre de lignes à lire du fichier CSV. Si None, tous les échantillons sont lus.
    nan_as_category : bool, optionnel
        Indique si les valeurs NaN doivent être traitées comme une catégorie lors de l'encodage One-Hot.
        Par défaut, True.

    Retours
    -------
    cc_agg : pandas.DataFrame
        DataFrame contenant les agrégations des soldes de cartes de crédit
        et le nombre de lignes de carte de crédit par client.

    Exemple
    -------
    >>> cc_agg = credit_card_balance(num_rows=1000, nan_as_category=True)
    """

    cc = df_credit_card_balance
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    #print(cc_agg.head())
    #print(cc_agg.isna().sum())
    cc_agg = cc_agg.astype(float)
    # Identifier les colonnes numériques
    col_num = cc_agg.select_dtypes(exclude = 'O').columns #exclude = 'O'; include = 'number'
    # print(col_num)
    # Calculer la moyenne des colonnes numériques
    moy = cc_agg[col_num].mean()
    # Remplacer les valeurs manquantes par la moyenne des colonnes numériques
    cc_agg[col_num] = cc_agg[col_num].fillna(moy)
    #print(cc_agg.isna().sum())
    cc_data = cc_agg
    return cc_data

def prep(app_data, df_bureau, df_bureau_balance, df_previous_application, df_POS_CASH_balance, df_installments_payments, df_credit_card_balance, verbose = True):
    debug = False
    num_rows = 10000 if debug else None

    # creer fonction prep_data, et fonction prediction (appel prep_data, sur le résultat elle applique le model juste en prediciton, et renvoie les prédictions)
    # output : dataframe enrichi de deux colonnes, prédict proba (proba associée à 1), et colonne binaire 0 ou 1 (accorder un crédit oui ou non)
    #   shap sur les prédictions
    # fonctions dans un autre notebook 

    # Charger et pré-traiter les données d'application
    df = application_train_test(app_data, num_rows)
    # Charger et pré-traiter les données de bureau et bureau_balance
    bureau = bureau_and_balance(df_bureau, df_bureau_balance, num_rows)
    if verbose : 
        print("Bureau df shape:", bureau.shape)
    # Joindre les données de bureau à df sur la clé 'SK_ID_CURR'
    df = df.join(bureau, how='left', on='SK_ID_CURR')

    # Charger et pré-traiter les données de previous_applications
    prev = previous_applications(df_previous_application, num_rows)
    if verbose :
        print("Previous applications df shape:", prev.shape)
    # Joindre les données de previous_applications à df sur la clé 'SK_ID_CURR'
    df = df.join(prev, how='left', on='SK_ID_CURR')

    # Charger et pré-traiter les données de pos_cash
    pos = pos_cash(df_POS_CASH_balance, num_rows)
    if verbose :
        print("Pos-cash balance df shape:", pos.shape)
    # Joindre les données de pos_cash à df sur la clé 'SK_ID_CURR'
    df = df.join(pos, how='left', on='SK_ID_CURR')

    # Charger et pré-traiter les données de installments_payments
    ins = installments_payments(df_installments_payments, num_rows)
    if verbose : 
        print("Installments payments df shape:", ins.shape)
    # Joindre les données de installments_payments à df sur la clé 'SK_ID_CURR'
    df = df.join(ins, how='left', on='SK_ID_CURR')

    # Charger et pré-traiter les données de credit_card_balance
    cc = credit_card_balance(df_credit_card_balance, num_rows)
    if verbose : 
        print("Credit card balance df shape:", cc.shape)
    # Joindre les données de credit_card_balance à df sur la clé 'SK_ID_CURR'
    df = df.join(cc, how='left', on='SK_ID_CURR')

    # Identifier les colonnes numériques
    #col_num = df.select_dtypes(include = 'number').columns #exclude = 'O'; include = 'number'
    # print(col_num)
    # Calculer la moyenne des colonnes numériques
    #moy = df[col_num].mean()
    # Remplacer les valeurs manquantes par la moyenne des colonnes numériques
    #df[col_num] = df[col_num].fillna(moy)

    #print(df.isna().sum())

    df_prep = clean_feature_names(df)

    df_prep.to_csv('../data/df_prep.csv')

    return df_prep

def prediction(df_prep, model_path = '../model/model.pkl'):
 
    # Charger le modèle sauvegardé
    filename = model_path
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Vérifier les données
    # Supposer que vous avez une colonne 'TARGET' dans df
    # Si la colonne 'TARGET' existe, supprimez-la (car vous n'en avez pas besoin pour les prédictions)
    if 'TARGET' in df_prep.columns:
        df_prep = df_prep.drop(columns=['TARGET'])
    
    # Faire des prédictions
    pred_proba = loaded_model.predict_proba(df_prep)[:, 1]  # Probabilité associée à la classe 1
    pred_binary = loaded_model.predict(df_prep)  # Prédictions binaires (0 ou 1)
    
    # Créer un DataFrame de résultats
    df_results = df_prep.copy()
    df_results['pred_proba'] = pred_proba
    df_results['pred_binary'] = pred_binary

    df_results.to_csv('../data/df_results.csv')
    
    return df_results
