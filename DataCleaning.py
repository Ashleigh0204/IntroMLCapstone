import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data():
    # load training data
    housing_trn_df = pd.read_csv("house-price-advanced-regression-techniques/train.csv")

    # split into training and validation
    housing_trn_df, housing_vld_df = train_test_split(housing_trn_df)

    # get target values
    housing_trn_target_ns = housing_trn_df['SalePrice'].to_numpy().reshape(-1,1)
    housing_vld_target_ns = housing_vld_df['SalePrice'].to_numpy().reshape(-1,1)
    housing_trn_df = housing_trn_df.drop(['SalePrice', 'Id'], axis=1)
    housing_vld_df = housing_vld_df.drop(['SalePrice', 'Id'], axis=1)
    
    # loading testing data
    housing_tst_df = pd.read_csv("house-price-advanced-regression-techniques/test.csv")
    housing_tst_df = housing_tst_df.drop('Id', axis=1)
    
    return housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns

def clean_features(housing_trn_df, housing_vld_df, housing_tst_df):
    column_names = housing_trn_df.columns
    categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'LandContour', 'Utilities', 'LotConfig', 'Neighborhood',
                    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                    'Foundation', 'Heating', 'Electrical', 'Functional', 'SaleType', 'SaleCondition']
    ordinal_cols = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                   'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC']
    fill_NA_cols = ['Alley', 'GarageType', 'Fence']
    fill_none_cols = ['MasVnrType', 'MiscFeature']
    fill_0_cols = ['LotFrontage', 'GarageYrBlt']
    numerical_cols = [col for col in column_names if col not in categorical_cols and col not in ordinal_cols and col not in fill_NA_cols and col not in fill_0_cols and col not in fill_none_cols]
    categories = [
        [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190],
        ['A', 'C (all)', 'FV', 'I', 'RH', 'RP', 'RM', 'RL'],
        ['Grvl', 'Pave'],
        ['Lvl', 'Bnk', 'HLS', 'Low'],
        ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
        ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
        ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
        ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
        ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'],
        ['1Fam', '2fmCon', 'Duplex','Twnhs', 'TwnhsE', 'TwnhsI'],
        ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
        ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
        ['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
        ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'],
        ['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shng'],
        ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
        ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
        ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
        ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
        ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'],
        ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
    ]
    other_cats = [
        ['Grvl', 'Pave', 'NA'],
        ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'],
        ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
    ]
    
    other_cats_none = [
            ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'],
            ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'None']
    ]
        
    ratings = [
        ['IR3', 'IR2', 'IR1', 'Reg'],
        ['Gtl', 'Mod', 'Sev'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'No', 'Mn', 'Av', 'Gd'],
        ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['N', 'Y'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Unf', 'RFn', 'Fin'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        ['N', 'P', 'Y'],
        ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    ]
    
    categorical_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='most_frequent')),
        ('one hot', OneHotEncoder(categories=categories, sparse_output=False))])
    
    NA_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='constant', fill_value='NA')),
        ('one hot', OneHotEncoder(categories=other_cats, sparse_output=False))])
    
    none_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='constant', fill_value='None')),
        ('one hot', OneHotEncoder(categories=other_cats_none, sparse_output=False))])
    
    num_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='mean')),
        ('standardize', StandardScaler())])
    
    zero_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='constant', fill_value=0)),
        ('standardize', StandardScaler())])
    
    ordinal_pipe = Pipeline([
        ('fill vals', SimpleImputer(strategy='constant', fill_value='NA')),
        ('ordinal', OrdinalEncoder(categories=ratings)),
        ('standardize', StandardScaler())])
    
    feature_processor = ColumnTransformer([
        ('categorical data', categorical_pipe, categorical_cols),
        ('NA', NA_pipe, fill_NA_cols),
        ('numerical data', num_pipe, numerical_cols),
        ('0', zero_pipe, fill_0_cols),
        ('ordinal data', ordinal_pipe, ordinal_cols)
    ])
    
    housing_trn_clean = feature_processor.fit_transform(housing_trn_df)
    housing_vld_clean = feature_processor.transform(housing_vld_df)
    housing_tst_clean = feature_processor.transform(housing_tst_df)
    
    return housing_trn_clean, housing_vld_clean, housing_tst_clean

def clean_target(housing_trn_target_ns, housing_vld_target_ns):
    target_pipe = Pipeline([('standardize', StandardScaler())])

    housing_trn_target = target_pipe.fit_transform(housing_trn_target_ns).reshape(-1)
    housing_vld_target = target_pipe.transform(housing_vld_target_ns).reshape(-1)

    return housing_trn_target, housing_vld_target