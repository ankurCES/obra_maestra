import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

nb_epochs = 100
batch_size = 5

def read_data(input_file, region_codes):
    df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
    area_frames = []
    for region_code in region_codes:
        df_region = df[df.region == region_code]
        area_frames.append(df_region)

    df_Area = pd.concat(area_frames)
    return df_Area

def generate_kill_plot(region_df):
    region_df.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,6),
                   s=region_df['nkill']*3, label= 'Casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    plt.savefig('plots/scatter_plot_region_casualities.png')

def generate_conf_matrix(region_df):
    corrmat = region_df.corr()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, vmax=1, square=True);
    plt.savefig('plots/confusion_matrix.png')

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Dense(12, input_dim=12, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def run_classifier(region_id):
    df_Area = read_data( 'globalterrorismdb_0718dist.csv', [region_id])

    df_Region = df_Area.drop([ 'region', 'claimed', 'nkillter', 'nwound','nwoundte'], axis=1)

    df_Region['nkill'].fillna(df_Region['nkill'].mean(), inplace=True)
    df_Region['latitude'].fillna(df_Region['latitude'].mean(), inplace=True)
    df_Region['longitude'].fillna(df_Region['longitude'].mean(), inplace=True)
    df_Region['natlty1'].fillna(df_Region['natlty1'].mean(), inplace=True)

    generate_kill_plot(df_Region)

    df_Region.corr()

    generate_conf_matrix(df_Region)


    X = df_Region.drop(['iyear', 'success','country', 'country_txt', 'attacktype1_txt','targtype1_txt','natlty1', 'natlty1_txt', 'weaptype1_txt'], axis=1)
    y = df_Region['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    criteria = [ 'imonth', 'iday', 'extended',  'latitude', 'longitude', 'multiple','suicide','attacktype1',
                'targtype1', 'individual', 'weaptype1', 'nkill']


    # y = df_Region['success']
    # X = df_Region[criteria]


    model = create_model(optimizer='adam', init='glorot_uniform')

    model.summary()

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    # Save entire model
    model.save('models/model_{}.h5'.format(region_id))
    print("Model Saved")

def predict_attack_success(attack_params, region_id):
    # Load the compiled model for the given region
    model = load_model('models/model_{}.h5'.format(region_id))

    outcome = model.predict(attack_params)
    print('Success rate of attack: {}%\n'.format(outcome[0][0]*100))

if __name__ == '__main__':
    region_id = 6
    # region id 6 --> India
    run_classifier(region_id)

    # Check globalterrorismdb_0718dist file for columns and mappings
    # Date params
    month = 6
    day = 14
    # Boolean 0-No,; 1-Yes
    extended = 0
    # Location
    latitude = 28.585836
    longitude = 77.153336
    # Attack Params
    multiple = 0
    suicide = 0
    attackType = 3
    targetType = 7
    individual = 0
    weaponType = 6
    # Aftermath --> Casuality Number
    nkill = 0

    attack_params = np.array([[(month),(day),(extended),(latitude),(longitude),(multiple),(suicide),(attackType),(targetType),(individual),(weaponType),(nkill)]])
    predict_attack_success(attack_params, region_id)
