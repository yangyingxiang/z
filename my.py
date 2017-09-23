import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import warnings
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

OUTLIER_THRES = 0.4

score = pd.read_csv('../input/train_2016_v2.csv')
features = pd.read_csv('../input/properties_2016.csv', low_memory=False)
train_labels = pd.read_excel('../input/zillow_data_dictionary.xlsx')

score = score.set_index('parcelid')
features = features.set_index('parcelid')

def get_continuous_feature(train_score_feature):
	continuous_variables = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
	continous_feature = train_score_feature[continuous_variables]
	# fill na for each variable
	# use taxvaluedollarcnt/ 73 to impute taxamount
	index = continous_feature['taxvaluedollarcnt'].notnull() & continous_feature['taxamount'].isnull()
	continous_feature.loc[index, 'taxamount'] = continous_feature[index]['taxvaluedollarcnt'] / 73
	# use taxvaluedollarcnt - landtaxvaluedollarcnt to impute structuretaxvaluedollarcnt
	index = continous_feature['taxvaluedollarcnt'].notnull() & continous_feature['landtaxvaluedollarcnt'].notnull() \
	& continous_feature['structuretaxvaluedollarcnt'].isnull()
	continous_feature.loc[index, 'structuretaxvaluedollarcnt'] = continous_feature[index]['taxvaluedollarcnt'] - \
	continous_feature[index]['landtaxvaluedollarcnt']
	# Use mean value to fill the rest
	continous_feature = continous_feature.fillna(continous_feature.mean())
	return continous_feature

def get_month_feature(train_score_feature):
	return pd.get_dummies(train_score_feature['transactiondate'].apply(lambda x: str(x).split('-')[1]))


def get_full_categorical_feature(train_score_feature):
	full_categorical_variables = ['airconditioningtypeid', 'heatingorsystemtypeid', 'fireplacecnt', 'garagecarcnt', 
                             'propertylandusetypeid', 'regionidcounty', 'unitcnt', 'buildingqualitytypeid', 'taxdelinquencyflag']
	binary_variables = ['garagetotalsqft']
	full_categorical_feature = train_score_feature[full_categorical_variables + binary_variables].fillna(-1).copy()
	for feature in binary_variables:
	    full_categorical_feature[full_categorical_feature[feature] > 0][feature] = 1
	full_categorical_one_hot_feature = pd.get_dummies(full_categorical_feature, columns=full_categorical_variables)
	return full_categorical_one_hot_feature

def get_selected_categorical_feature(train_score_feature):
	# variables with selected categories
	selected_categorical_variables = ['propertyzoningdesc', 'regionidneighborhood', 'regionidzip']
	selected_categorical_feature = train_score_feature[selected_categorical_variables].fillna(-1)
	feature_names = ['propertyzoningdesc_CA&MUR&D*', 'propertyzoningdesc_RBR4YY', 'propertyzoningdesc_SDSF7500RP', 'propertyzoningdesc_LRR10000*', 'propertyzoningdesc_WVRPD12U-R', 'propertyzoningdesc_PDR16000*', 'propertyzoningdesc_LBCCA', 'propertyzoningdesc_SDSF7500*', 'propertyzoningdesc_LCR3800030', 'propertyzoningdesc_WHR172', 'propertyzoningdesc_SGR17500SI', 'propertyzoningdesc_WCR19450*', 'propertyzoningdesc_SLR3*', 'propertyzoningdesc_CVR1*', 'propertyzoningdesc_PDSP*', 'propertyzoningdesc_BFA1*', 'regionidneighborhood_416343.0', 'regionidneighborhood_764108.0', 'regionidneighborhood_764105.0', 'regionidneighborhood_763219.0', 'regionidneighborhood_275129.0', 'regionidneighborhood_762684.0', 'regionidneighborhood_275916.0', 'regionidneighborhood_417225.0', 'regionidneighborhood_763171.0', 'regionidneighborhood_275979.0', 'regionidneighborhood_265889.0', 'regionidneighborhood_762177.0', 'regionidneighborhood_417433.0', 'regionidneighborhood_623378.0', 'regionidneighborhood_268316.0', 'regionidneighborhood_763220.0', 'regionidneighborhood_268132.0', 'regionidneighborhood_275795.0', 'regionidneighborhood_275795.0', 'regionidzip_96943.0', 'regionidzip_96951.0', 'regionidzip_96951.0']
	selected_categorical_one_hot_feature = pd.get_dummies(selected_categorical_feature, columns=selected_categorical_variables)[feature_names]
	return selected_categorical_one_hot_feature


def add_month_feature_for_prediction(df, month):
	month_columns = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
	for m in range(1, 13):
		if m == month:
			df[month_columns[m]] = 1
		else:
			df[month_columns[m]] = 0
	return df


continous_feature = get_continuous_feature(features)
full_categorical_one_hot_feature = get_full_categorical_feature(features)
# selected_categorical_one_hot_feature = get_selected_categorical_feature(features)

# train_score_feature = pd.concat([score, features], axis=1, join_axes=[score.index])
combined_feature = pd.concat([continous_feature, full_categorical_one_hot_feature], axis=1)
train_score_feature = pd.concat([score, combined_feature], axis=1, join_axes=[score.index])
# Filter out outlier
train_score_feature = train_score_feature[(train_score_feature['logerror'] < OUTLIER_THRES) & (train_score_feature['logerror'] > -1 * OUTLIER_THRES)]
train_score_feature = train_score_feature.sample(frac=1).reset_index(drop=True)

month_feature = get_month_feature(train_score_feature)
train_feature = pd.concat([month_feature, train_score_feature], axis=1, join_axes=[train_score_feature.index])


# Train the model
y = train_score_feature['logerror']
train_feature.drop(['logerror', 'transactiondate'], axis=1, inplace=True)
model = RandomForestRegressor(n_jobs=10, criterion='mae', n_estimators=200, max_features=100, verbose=True)
model.fit(train_feature, y); print('fit...')
print(mean_absolute_error(y, reg.predict(train)))

# Make prediction
print('Make predictions')
result = pd.DataFrame()
result['ParcelId'] = combined_feature.index
for month in [10, 11, 12]:
	print(month)
	preict_feature = add_month_feature_for_prediction(combined_feature, month)
	y_pred = model.predict(preict_feature)
	result[month] = y_pred
result.columns = ['ParcelId', '201610', '201611', '201612']
result['201710'] = 0
result['201711'] = 0
result['201712'] = 0
# Write result into csv
result.to_csv('result.csv')


