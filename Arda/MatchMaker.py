import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from helper_funcs import normalize


def data_loader(drug1_chemicals,drug2_chemicals,cell_line_gex,comb_data_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")
    cell_line = pd.read_csv(cell_line_gex,header=None)
    chem1 = pd.read_csv(drug1_chemicals,header=None)
    chem2 = pd.read_csv(drug2_chemicals,header=None)
    synergies = np.array(comb_data["synergy_loewe"])

    cell_line = np.array(cell_line.values)
    chem1 = np.array(chem1.values)
    chem2 = np.array(chem2.values)
    return chem1, chem2, cell_line, synergies


def prepare_data(chem1, chem2, cell_line, synergies, norm, train_ind_fname, val_ind_fname, test_ind_fname):
    print("Data normalization and preparation of train/validation/test data")
    test_ind = list(np.loadtxt(test_ind_fname, dtype=int))
    val_ind = list(np.loadtxt(val_ind_fname, dtype=int))
    train_ind = list(np.loadtxt(train_ind_fname, dtype=int))

    train_data = {}
    val_data = {}
    test_data = {}

    if norm == "minmax":
        # ---------- drug1 ----------
        train1 = np.concatenate((chem1[train_ind, :], chem2[train_ind, :]), axis=0)
        train_data['drug1'], mins1, maxs1, feat_filt1 = normalize(train1, norm='minmax')
        val_data['drug1'], _, _, _ = normalize(chem1[val_ind, :], norm='minmax', mins=mins1, maxs=maxs1, feat_filt=feat_filt1)
        test_data['drug1'], _, _, _ = normalize(chem1[test_ind, :], norm='minmax', mins=mins1, maxs=maxs1, feat_filt=feat_filt1)

        # ---------- drug2 ----------
        train2 = np.concatenate((chem2[train_ind, :], chem1[train_ind, :]), axis=0)
        train_data['drug2'], mins2, maxs2, feat_filt2 = normalize(train2, norm='minmax')
        val_data['drug2'], _, _, _ = normalize(chem2[val_ind, :], norm='minmax', mins=mins2, maxs=maxs2, feat_filt=feat_filt2)
        test_data['drug2'], _, _, _ = normalize(chem2[test_ind, :], norm='minmax', mins=mins2, maxs=maxs2, feat_filt=feat_filt2)

        # ---------- cell line ----------
        train3 = np.concatenate((cell_line[train_ind, :], cell_line[train_ind, :]), axis=0)
        train_cell_line, mins3, maxs3, feat_filt3 = normalize(train3, norm='minmax')
        val_cell_line, _, _, _ = normalize(cell_line[val_ind, :], norm='minmax', mins=mins3, maxs=maxs3, feat_filt=feat_filt3)
        test_cell_line, _, _, _ = normalize(cell_line[test_ind, :], norm='minmax', mins=mins3, maxs=maxs3, feat_filt=feat_filt3)

    else:
        # ---------- existing tanh_norm / norm / tanh behavior ----------
        train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0)
        train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
        val_data['drug1'], _, _, _, _, _ = normalize(chem1[val_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
        test_data['drug1'], _, _, _, _, _ = normalize(chem1[test_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

        train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
        train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
        val_data['drug2'], _, _, _, _, _ = normalize(chem2[val_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
        test_data['drug2'], _, _, _, _, _ = normalize(chem2[test_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

        train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
        train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
        val_cell_line, _, _, _, _, _ = normalize(cell_line[val_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
        test_cell_line, _, _, _, _, _ = normalize(cell_line[test_ind,:], mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

    train_data['drug1'] = np.concatenate((train_data['drug1'], train_cell_line), axis=1)
    train_data['drug2'] = np.concatenate((train_data['drug2'], train_cell_line), axis=1)

    val_data['drug1'] = np.concatenate((val_data['drug1'], val_cell_line), axis=1)
    val_data['drug2'] = np.concatenate((val_data['drug2'], val_cell_line), axis=1)

    test_data['drug1'] = np.concatenate((test_data['drug1'], test_cell_line), axis=1)
    test_data['drug2'] = np.concatenate((test_data['drug2'], test_cell_line), axis=1)

    train_data['y'] = np.concatenate((synergies[train_ind], synergies[train_ind]), axis=0)
    val_data['y'] = synergies[val_ind]
    test_data['y'] = synergies[test_ind]

    # safety
    for split_data in [train_data, val_data, test_data]:
        split_data["drug1"] = np.nan_to_num(split_data["drug1"], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        split_data["drug2"] = np.nan_to_num(split_data["drug2"], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        split_data["y"] = np.nan_to_num(split_data["y"], nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    print(test_data['drug1'].shape)
    print(test_data['drug2'].shape)
    return train_data, val_data, test_data

def generate_network(train, layers, inDrop, drop):
    # fill the architecture params from dict
    dsn1_layers = str(layers["DSN_1"]).split("-")
    dsn2_layers = str(layers["DSN_2"]).split("-")
    snp_layers = layers["SPN"].split("-")
    # contruct two parallel networks
    for l in range(len(dsn1_layers)):
        if l == 0:
            input_drug1    = Input(shape=(train["drug1"].shape[1],))
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug1)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn1_layers)-1):
            dsn1_output = Dense(int(dsn1_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    for l in range(len(dsn2_layers)):
        if l == 0:
            input_drug2    = Input(shape=(train["drug2"].shape[1],))
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug2)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn2_layers)-1):
            dsn2_output = Dense(int(dsn2_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)
    
    concatModel = concatenate([dsn1_output, dsn2_output])
    
    for snp_layer in range(len(snp_layers)):
        if len(snp_layers) == 1:
            snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
            snp_output = Dense(1, activation='linear')(snpFC)
        else:
            # more than one FC layer at concat
            if snp_layer == 0:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
                snpFC = Dropout(float(drop))(snpFC)
            elif snp_layer == (len(snp_layers)-1):
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snp_output = Dense(1, activation='linear')(snpFC)
            else:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snpFC = Dropout(float(drop))(snpFC)

    model = Model([input_drug1, input_drug2], snp_output)
    return model

def trainer(model, l_rate, train, val, epo, batch_size, earlyStop, modelName, weights, use_wandb=False):
    callbacks = [
        ModelCheckpoint(modelName, verbose=1, monitor='val_loss', save_best_only=True, mode='auto'),
        EarlyStopping(monitor='val_loss', mode='auto', patience=earlyStop, restore_best_weights=False)
    ]

    if use_wandb:
        try:
            from wandb.integration.keras import WandbMetricsLogger
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        except Exception:
            pass

    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(
            learning_rate=float(l_rate),
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False
        )
    )

    model.fit(
        [train["drug1"], train["drug2"]],
        train["y"],
        epochs=epo,
        shuffle=True,
        batch_size=batch_size,
        verbose=1,
        validation_data=([val["drug1"], val["drug2"]], val["y"]),
        sample_weight=weights,
        callbacks=callbacks
    )

    return model

def predict(model, data):
    pred = model.predict(data)
    return pred.flatten()
