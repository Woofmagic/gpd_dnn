
import glob
import gc
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import corner
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import gepard as g
from gepard.fits import th_KM15
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

from simultaneous_fit_dnn_config import compute_fe
from simultaneous_fit_dnn_config import compute_fg
from simultaneous_fit_dnn_config import compute_f2
from simultaneous_fit_dnn_config import compute_f1
from simultaneous_fit_dnn_config import compute_epsilon
from simultaneous_fit_dnn_config import compute_y
from simultaneous_fit_dnn_config import compute_skewness
from simultaneous_fit_dnn_config import compute_t_min
from simultaneous_fit_dnn_config import compute_t_prime
from simultaneous_fit_dnn_config import compute_k_tilde
from simultaneous_fit_dnn_config import compute_k
from simultaneous_fit_dnn_config import compute_k_dot_delta
from simultaneous_fit_dnn_config import prop_1
from simultaneous_fit_dnn_config import prop_2

from simultaneous_fit_dnn_config import bkm10_cross_section
from simultaneous_fit_dnn_config import bkm10_bsa

with open("closure_test_config.yml", "r") as file:
    config = yaml.safe_load(file)

MAJOR_NUMBER = config["versioning"]["major"]
MINOR_NUMBER = config["versioning"]["minor"]
MAJOR_MINOR_NUMBER = f"{MAJOR_NUMBER}_{MINOR_NUMBER}"

NUMBER_OF_EPOCHS = config["dnn_config"]["epochs"]
NUMBER_OF_REPLICAS = config["dnn_config"]["replicas"]
BATCH_SIZE = config["dnn_config"]["batch_size"]
LEARNING_RATE = config["dnn_config"]["adam_learning_rate"]

IS_CFF_REAL_H_FREE = config["cff_config"]["enable_cff_real_h"]
IS_CFF_IMAG_H_FREE = config["cff_config"]["enable_cff_imag_h"]
IS_CFF_REAL_HT_FREE = config["cff_config"]["enable_cff_real_ht"]
IS_CFF_IMAG_HT_FREE = config["cff_config"]["enable_cff_imag_ht"]
IS_CFF_REAL_E_FREE = config["cff_config"]["enable_cff_real_e"]
IS_CFF_IMAG_E_FREE = config["cff_config"]["enable_cff_imag_e"]
IS_CFF_REAL_ET_FREE = config["cff_config"]["enable_cff_real_et"]
IS_CFF_IMAG_ET_FREE = config["cff_config"]["enable_cff_imag_et"]

# cross-section observables:
IS_UNP_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_unp_target_xsec"]
IS_PLUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_unp_target_xsec"]
IS_MINUS_BEAM_UNP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_unp_target_xsec"]
IS_UNP_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_unp_beam_lp_target_xsec"]
IS_PLUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_plus_beam_lp_target_xsec"]
IS_MINUS_BEAM_LP_TARGET_XSEC_INCLUDED = config["observable_config"]["enable_minus_beam_lp_target_xsec"]

IS_UNP_TARGET_BSA_INCLUDED = config["observable_config"]["enable_unp_target_bsa"]
IS_PLUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_plus_target_bsa"]
IS_MINUS_TARGET_BSA_INCLUDED = config["observable_config"]["enable_minus_target_bsa"]

IS_UNP_BEAM_TSA_INCLUDED = config["observable_config"]["enable_unp_beam_tsa"]
IS_PLUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_plus_beam_tsa"]
IS_MINUS_BEAM_TSA_INCLUDED = config["observable_config"]["enable_minus_beam_tsa"]

IS_DSA_INCLUDED = config["observable_config"]["enable_dsa"]

print(f"[INFO]: Recieved major version number: {MAJOR_NUMBER}")
print(f"[INFO]: Recieved minor version number: {MINOR_NUMBER}")
print(f"[INFO]: Recieved total version number: {MAJOR_MINOR_NUMBER}")

print(f"[INFO]: Received number of epochs (per replica): {NUMBER_OF_EPOCHS}")
print(f"[INFO]: Received number of replicas: {NUMBER_OF_REPLICAS}")
print(f"[INFO]: Received batch size: {BATCH_SIZE}")
print(f"[INFO]: Received (Adam) learning rate value: {LEARNING_RATE}")

STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360
NUMBER_OF_PHI_POINTS = 100

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = [np.radians(degree_value) for degree_value in phi_array_in_degrees]

print(
    f"[INFO]: New list of {len(phi_array_in_radians)} of azimuthal angles "
    f"from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees")

FIXED_K = 5.750
FIXED_XB = 0.360
FIXED_T = -0.17
FIXED_Q_SQUARED = 2.300

try:
    # [NOTE]: We actually don't need to be super accurate here because we ONLY use
    # this class to evaluate the CFFs later!
    test_datapoints = [g.DataPoint(
        xB = FIXED_XB, t = FIXED_T, Q2 = FIXED_Q_SQUARED, phi = fixed_phi,
        process = "ep2epgamma", exptype = 'fixed target',
        in1energy = FIXED_K, in1charge = -1, in1polarization = +1, in1units = 'rad',
        observable = 'XS',
        fname = 'Trento') for fixed_phi in phi_array_in_radians]
except ZeroDivisionError:
    print(f"[ERROR]: Kinematic setting k = {FIXED_K}, xb = {FIXED_XB}, t = {FIXED_T}, Q^2 = {FIXED_Q_SQUARED} unphysical according to gepard.")

# here we compute the actual CFFs!
real_h_values = np.array([th_KM15.ReH(datapoint) for datapoint in test_datapoints])
imag_h_values = np.array([th_KM15.ImH(datapoint) for datapoint in test_datapoints])
real_e_values = np.array([th_KM15.ReE(datapoint) for datapoint in test_datapoints])
imag_e_values = np.array([th_KM15.ImE(datapoint) for datapoint in test_datapoints])
real_ht_values = np.array([th_KM15.ReHt(datapoint) for datapoint in test_datapoints])
imag_ht_values = np.array([th_KM15.ImHt(datapoint) for datapoint in test_datapoints])
real_et_values = np.array([th_KM15.ReEt(datapoint) for datapoint in test_datapoints])
imag_et_values = np.array([th_KM15.ImEt(datapoint) for datapoint in test_datapoints])

# here we actually set the KM15 values to 0:
CFF_REAL_H_KM15 = real_h_values[0] if IS_CFF_REAL_H_FREE else 0.0
CFF_IMAG_H_KM15 = imag_h_values[0] if IS_CFF_IMAG_H_FREE else 0.0
CFF_REAL_HT_KM15 = real_ht_values[0] if IS_CFF_REAL_HT_FREE else 0.0
CFF_IMAG_HT_KM15 = imag_ht_values[0] if IS_CFF_IMAG_HT_FREE else 0.0
CFF_REAL_E_KM15 = real_e_values[0] if IS_CFF_REAL_E_FREE else 0.0
CFF_IMAG_E_KM15 = imag_e_values[0] if IS_CFF_IMAG_E_FREE else 0.0
CFF_REAL_ET_KM15 = real_et_values[0] if IS_CFF_REAL_ET_FREE else 0.0
CFF_IMAG_ET_KM15 = imag_et_values[0] if IS_CFF_IMAG_ET_FREE else 0.0

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)

this_kinematic_set_title_string = (
    rf"$k = {FIXED_K:.3f}$ GeV, "
    rf"$x_B = {FIXED_XB:.3f}$, "
    rf"$t = {FIXED_T:.3f}$ GeV$^2$, "
    rf"$Q^2 = {FIXED_Q_SQUARED:.3f}$ GeV$^2$"
)

km15_cross_section = DifferentialCrossSection(
    configuration = {
        "kinematics_and_phi": BKM10Inputs(
            lab_kinematics_k = FIXED_K,
            squared_Q_momentum_transfer = FIXED_Q_SQUARED,
            x_Bjorken = FIXED_XB,
            squared_hadronic_momentum_transfer_t = FIXED_T),
        "cff_inputs": CFFInputs(
            compton_form_factor_h = CFF_H_KM15,
            compton_form_factor_h_tilde = CFF_H_TILDE_KM15,
            compton_form_factor_e = CFF_E_KM15,
            compton_form_factor_e_tilde = CFF_E_TILDE_KM15),
        "using_ww": True
    },
    verbose = False, debugging = False)
 
bkm10_unp_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

bkm10_bsa_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = 0.0).real

k = np.full(NUMBER_OF_PHI_POINTS, FIXED_K, dtype = np.float32)
t = np.full(NUMBER_OF_PHI_POINTS, FIXED_T, dtype = np.float32)
xb = np.full(NUMBER_OF_PHI_POINTS, FIXED_XB, dtype = np.float32)
q2 = np.full(NUMBER_OF_PHI_POINTS, FIXED_Q_SQUARED, dtype = np.float32)

phi = np.array(phi_array_in_radians, dtype = np.float32)

fe = compute_fe(t)
fg = compute_fg(fe)
f2 = compute_f2(t, fe, fg)
f1 = compute_f1(fg, f2)

epsilon = compute_epsilon(xb,q2)
y = compute_y(k, q2, epsilon)
xi = compute_skewness(xb, t, q2)
t_min = compute_t_min(xb, q2, epsilon)
t_prime = compute_t_prime(t, t_min)
k_tilde = compute_k_tilde(xb, q2, t, t_min, epsilon)
kinematic_k = compute_k(q2, y, epsilon, k_tilde)

k_dot_delta = compute_k_dot_delta(
    q2, xb, t, phi,
    epsilon, y, kinematic_k)
prop_1_values = prop_1(q2, k_dot_delta)
prop_2_values = prop_2(q2, t, k_dot_delta)

kinematics_and_phi = np.column_stack((t, xb, q2, phi, )).astype(np.float32)

physics_data = np.column_stack((
    # kinematics, phi:
    t, xb, q2, phi,

    # form factors:
    fe, fg, f1, f2,

    # derived kinematics:
    epsilon, y, xi, t_prime, k_tilde, kinematic_k,

    # phi-dependent stuff:
    k_dot_delta,  prop_1_values, prop_2_values,
)).astype(np.float32)

# stack the observables to fit:
observables = np.column_stack((
    bkm10_unp_beam_unp_target_km15, bkm10_bsa_km15,
)).astype(np.float32)

indices = np.arange(NUMBER_OF_PHI_POINTS)

dnn_inputs = np.column_stack((t, xb, q2, )).astype(np.float32)

training_indices, testing_indices = train_test_split(
    indices, test_size = 0.10, random_state = 7009,)

training_indices, validation_indices = train_test_split(
    training_indices, test_size = 0.10, random_state = 7009,)

splits = {
    "train": training_indices,
    "validation": validation_indices,
    "test": testing_indices,
}

x_training = dnn_inputs[training_indices]
x_validation = dnn_inputs[validation_indices]
x_testing = dnn_inputs[testing_indices]

precomputed_physics_training = physics_data[training_indices]
precomputed_physics_validation = physics_data[validation_indices]
precomputed_physics_testing = physics_data[testing_indices]

y_training = observables[training_indices]
y_validation = observables[validation_indices]
y_testing = observables[testing_indices]

# npz nowadays:
np.savez(
    'testing_npz_rightnow',
    x_training = x_training,
    x_validation= x_validation,
    x_testing = x_testing,

    physics_training = precomputed_physics_training,
    physics_validation = precomputed_physics_validation,
    physics_testing = precomputed_physics_testing,

    y_training = y_training,
    y_validation = y_validation,
    y_testing = y_testing,

    training_indices = training_indices,
    validation_indices = validation_indices,
    testing_indices = testing_indices,
)

class SimultaneousObservablesLoss(tf.keras.losses.Loss):
    def __init__(self, name = "simultaneous_observables_loss"):
        super().__init__(name = name)

        self._OBSERVABLE_WEIGHT_1 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_2 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_3 = 0.0 * 0.5
        self._OBSERVABLE_WEIGHT_4 = 0.0 * 0.5
    
    @tf.function
    def call(self, true_values, predicted_values):
        
        # the CFFs:
        cff_h_real_tf = predicted_values[:, 0]
        cff_h_imag_tf = predicted_values[:, 1]
        cff_ht_real_tf = predicted_values[:, 2]
        cff_ht_imag_tf = predicted_values[:, 3]

        # kinematics_and_phi:
        t_tf = predicted_values[:, 4]
        xb_tf = predicted_values[:, 5]
        q_squared_tf = predicted_values[:, 6]
        phi_tf = predicted_values[:, 7]

        # derived quantities -> form factors:
        fe_tf = predicted_values[:, 8]
        fg_tf = predicted_values[:, 9]
        f1_tf = predicted_values[:, 10]
        f2_tf = predicted_values[:, 11]

        # derived quantities -> kinematics_and_phi
        epsilon_tf = predicted_values[:, 12]
        y_lep_tf = predicted_values[:, 13]
        xi_tf = predicted_values[:, 14]
        tprime_tf = predicted_values[:, 15]
        ktilde_tf = predicted_values[:, 16]
        k_tf  = predicted_values[:, 17]

        # derived quantities -> phi-depdendent stuff:
        kdd_tf = predicted_values[:, 18]
        p1_tf = predicted_values[:, 19]
        p2_tf = predicted_values[:, 20]

        # observables:
        true_cross_section = true_values[:, 0]
        true_bsa = true_values[:, 1]

        cross_section = bkm10_cross_section(
            0.0, 0.0,
            q_squared_tf, xb_tf, t_tf, epsilon_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_tf, p1_tf, p2_tf,
            cff_h_real_tf, cff_ht_real_tf, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag_tf, cff_ht_imag_tf, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)
        
        # compute cross-section residuals:
        residuals_cross_section = true_cross_section - cross_section

        predicted_bsa = bkm10_bsa(
            0.0,
            q_squared_tf, xb_tf, t_tf, epsilon_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_tf, p1_tf, p2_tf,
            cff_h_real_tf, cff_ht_real_tf, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag_tf, cff_ht_imag_tf, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)
        
        # compute BSA residuals:
        residuals_bsa = true_bsa - predicted_bsa

        # compute the MSE:
        mean_squared_error = (
            self._OBSERVABLE_WEIGHT_1 * tf.reduce_mean(tf.square(residuals_cross_section))+
            self._OBSERVABLE_WEIGHT_2 * tf.reduce_mean(tf.square(residuals_bsa)))

        return mean_squared_error

def cff_h_model():

    kinematics_inputs = tf.keras.Input(shape = (3,), name = "input_values")
    physics_input = tf.keras.Input(shape = (17,), name = "precomputed_physics")
    dnn_kinematic_inputs = tf.keras.layers.Lambda(lambda x: x[:, :3], name = "input_kinematics")(kinematics_inputs)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = "he_normal", activation = "relu")(dnn_kinematic_inputs)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = "he_normal", activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = "he_normal", activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = "he_normal", activation = "relu")(hidden)
    cff_outputs = tf.keras.layers.Dense(4, activation = "linear", name = "cff_h_htilde")(hidden) # Re[H], Im[H], Re[Ht], Im[Ht]
    full_model_outputs = tf.keras.layers.Concatenate(name = "physics_and_cffs")([cff_outputs, physics_input])
    model = tf.keras.Model(inputs = [kinematics_inputs, physics_input], outputs = full_model_outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(lening_rate = LEARNING_RATE),
        loss = SimultaneousObservablesLoss(),
        jit_compile = True,
        )
    
    # return model:
    return model

for replica in range(NUMBER_OF_REPLICAS):

    tf.keras.backend.clear_session()
    gc.collect()

    replica_number = replica + 1

    dnn_model = cff_h_model()

    dnn_model_history = dnn_model.fit(
        x = {
            "input_values": x_training,
            "precomputed_physics": precomputed_physics_training
        },
        y = y_training,
        validation_data = (
            {
                "input_values": x_validation,
                "precomputed_physics": precomputed_physics_validation
            },
            y_validation
        ),
        epochs = NUMBER_OF_EPOCHS,
        batch_size = BATCH_SIZE,
        verbose = 1
        )

    number_of_epochs_run = len(dnn_model_history.epoch)
    print(f"The model ran for {number_of_epochs_run} epochs before early stopping.")

    dnn_model.save(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")

    training_loss_data = dnn_model_history.history["loss"]
    validation_loss_data = dnn_model_history.history["val_loss"]

    dnn_evaluation_statistics = dnn_model.evaluate(
        x = {
            "input_values": x_testing,
            "precomputed_physics": precomputed_physics_testing,
        },
        y = y_testing,
        verbose = 0)
    
    print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")

    # make DF with testing metrics:
    pd.DataFrame({
        'testing_loss': [dnn_evaluation_statistics], # https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!
    }).to_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_loss_data.csv", 
        index = False)

    # save npz with DNN training information:
    np.savez(
        file = f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_losses_vs_epochs.npz",
        training_loss = training_loss_data,
        validation_loss = validation_loss_data
        )

    # cleanup
    del dnn_model

loss_files = sorted(glob.glob(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_*_losses_vs_epochs.npz"))

for replica_index, loss_file in enumerate(loss_files, start = 1):

    loss_information = np.load(loss_file)

    training_loss_data = loss_information["training_loss"]
    validation_loss_data = loss_information["validatioarn_loss"]

    number_of_epochs_run = len(training_loss_data)
    epochs = np.arange(number_of_epochs_run)

    testing_information = pd.read_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_index}_loss_data.csv"
    )

    testing_loss = testing_information["testing_loss"].iloc[0]

    curves_fig, curves_ax = plt.subplots(1, figsize = (8, 8))
    log_curves_fig, log_curves_ax = plt.subplots(1, figsize = (8, 8))

    initial_loss_value = training_loss_data[0]

    curves_ax.axhline(initial_loss_value, color = "red", linestyle = "--", label = "Initial Loss Value")
    curves_ax.axhline(0.0, color = "green", linestyle="--", label = r"Loss$ = 0$")

    curves_ax.plot(np.arange(0, number_of_epochs_run, 1), training_loss_data, color = "blue", label = "Training Loss")
    curves_ax.plot(np.arange(0, number_of_epochs_run, 1), validation_loss_data, color = "purple", label = "Validation Loss")

    log_curves_ax.axhline(initial_loss_value, color = "red", linestyle = "--", label = "Initial Loss Value")
    log_curves_ax.axhline(0.0, color = "green", linestyle = "--", label = r"Loss$ = 0$")

    # just do this for now hahahaha:
    _REGULARIZER = 1e-21

    log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(training_loss_data + _REGULARIZER), color = "blue", label = "Log Training Loss")
    log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(validation_loss_data + _REGULARIZER), color = "purple", label = "Log Validation Loss")

    curves_ax.legend(fontsize = 15)
    log_curves_ax.legend(fontsize = 15)

    curves_ax.set_xlabel("Epoch", fontsize = 15)
    curves_ax.set_ylabel("MSE", fontsize = 15)
    curves_ax.set_title(f"Replica {replica + 1} Learning Curves\n(Eval. Loss $= {testing_loss:.3g}$", fontsize = 16.)

    log_curves_ax.set_xlabel("Epoch", fontsize = 15)
    log_curves_ax.set_ylabel("Log MSE Loss", fontsize = 15)
    log_curves_ax.set_title(f"Replica {replica + 1} Learning Curves\n(Eval. Loss $= {testing_loss:.3g}$", fontsize = 15.)

    curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/lc_replica_{replica + 1}_v{MAJOR_MINOR_NUMBER}.png")
    curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/lc_replica_{replica + 1}_v{MAJOR_MINOR_NUMBER}.eps")

    log_curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/log_lc_replica_{replica + 1}_v{MAJOR_MINOR_NUMBER}.png")
    log_curves_fig.savefig(f"./local/version_{MAJOR_MINOR_NUMBER}/learning_curves/log_lc_replica_{replica + 1}_v{MAJOR_MINOR_NUMBER}.eps")

    plt.close(curves_fig)
    plt.close(log_curves_fig)

    del curves_fig
    del log_curves_fig

range_of_t = np.linspace(x_training["t"].min(), x_training["t"].max())
range_of_x_b = np.linspace(x_training["x_b"].min(), x_training["x_b"].max())
range_of_q_squared = np.linspace(x_training["q_squared"].min(), x_training["q_squared"].max())

replica_paths = sorted(glob.glob(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_*_v{MAJOR_MINOR_NUMBER}.keras"))
replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]
print(f"[INFO]: Loaded {len(replicas)} replica models.")

all_predictions = []

for replica in replicas:
    
    predicted_outputs = replica.predict(
        {
            "input_values": dnn_inputs,
            "precomputed_physics": physics_data,
        },
        verbose = 0,
    ) # predicting using x_data
    all_predictions.append(predicted_outputs)

all_predictions = np.array(all_predictions)

replicas_cross_predictions = []
replicas_bsa_predictions = []

for index, _ in enumerate(all_predictions):

    prediction = all_predictions[index]

    cff_h_real = prediction[:, 0]
    cff_h_imag = prediction[:, 1]
    cff_ht_real = prediction[:, 2]
    cff_ht_imag = prediction[:, 3]

    t = prediction[:, 4]
    xb = prediction[:, 5]
    q_squared = prediction[:, 6]
    phi = prediction[:, 7]

    fe = prediction[:, 8]
    fg = prediction[:, 9]
    f1 = prediction[:,10]
    f2 = prediction[:,11]

    epsilon = prediction[:,12]
    y_lep = prediction[:,13]
    xi = prediction[:,14]
    tprime = prediction[:,15]
    ktilde = prediction[:,16]
    k = prediction[:,17]

    kdd = prediction[:,18]
    p1 = prediction[:,19]
    p2 = prediction[:,20]

    cross_section = bkm10_cross_section(
        0.0, 0.0,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    predicted_bsa = bkm10_bsa(
        0.0,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    replicas_cross_predictions.append(cross_section)
    replicas_bsa_predictions.append(predicted_bsa)

replicas_cross_predictions = np.array(replicas_cross_predictions)
replicas_bsa_predictions = np.array(replicas_bsa_predictions)

mean_xs = np.mean(replicas_cross_predictions, axis = 0)
std_dev_xs = np.std(replicas_cross_predictions, axis = 0)

xs_mean = np.mean(replicas_cross_predictions, axis = 0)
xs_min = np.min(replicas_cross_predictions, axis = 0)
xs_max = np.max(replicas_cross_predictions, axis = 0)
xs_q1 = np.percentile(replicas_cross_predictions, 25, axis = 0)
xs_q3 = np.percentile(replicas_cross_predictions, 75, axis = 0)

xs_percentile_10 = np.percentile(replicas_cross_predictions, 10, axis = 0)
xs_percentile_20 = np.percentile(replicas_cross_predictions, 20, axis = 0)
xs_percentile_30 = np.percentile(replicas_cross_predictions, 30, axis = 0)
xs_percentile_40 = np.percentile(replicas_cross_predictions, 40, axis = 0)
xs_median = np.percentile(replicas_cross_predictions, 50, axis = 0)
xs_percentile_60 = np.percentile(replicas_cross_predictions, 60, axis = 0)
xs_percentile_70 = np.percentile(replicas_cross_predictions, 70, axis = 0)
xs_percentile_80 = np.percentile(replicas_cross_predictions, 80, axis = 0)
xs_percentile_90 = np.percentile(replicas_cross_predictions, 90, axis = 0)

mean_bsa = np.mean(replicas_bsa_predictions, axis = 0)
std_dev_bsa = np.std(replicas_bsa_predictions, axis = 0)

bsa_mean = np.mean(replicas_bsa_predictions, axis = 0)
bsa_min = np.min(replicas_bsa_predictions, axis = 0)
bsa_max = np.max(replicas_bsa_predictions, axis = 0)
bsa_q1 = np.percentile(replicas_bsa_predictions, 25, axis = 0)
bsa_q3 = np.percentile(replicas_bsa_predictions, 75, axis = 0)

bsa_percentile_10 = np.percentile(replicas_bsa_predictions, 10, axis = 0)
bsa_percentile_20 = np.percentile(replicas_bsa_predictions, 20, axis = 0)
bsa_percentile_30 = np.percentile(replicas_bsa_predictions, 30, axis = 0)
bsa_percentile_40 = np.percentile(replicas_bsa_predictions, 40, axis = 0)
bsa_median = np.percentile(replicas_bsa_predictions, 50, axis = 0)
bsa_percentile_60 = np.percentile(replicas_bsa_predictions, 60, axis = 0)
bsa_percentile_70 = np.percentile(replicas_bsa_predictions, 70, axis = 0)
bsa_percentile_80 = np.percentile(replicas_bsa_predictions, 80, axis = 0)
bsa_percentile_90 = np.percentile(replicas_bsa_predictions, 90, axis = 0)

fig2, ax2 = plt.subplots(1, figsize = (10, 7))

ax2.scatter(
    phi_array_in_radians, bkm10_unp_beam_unp_target_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax2.plot(
    phi_array_in_radians,
    mean_xs,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_max,
    y2 = xs_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_90,
    y2 = xs_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_80,
    y2 = xs_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_70,
    y2 = xs_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_60,
    y2 = xs_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax2.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax2.set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16)
ax2.set_title(
    rf"$d^{{4}}\sigma^{{UU}}$ vs. $\phi$, {this_kinematic_set_title_string}"
    "\n"
    f"(KM15): {km15_cff_string}", fontsize = 16
)

ax2.legend()
plt.tight_layout()

replica_cross_section_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/dnn_xsec_vs_phi_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig2.savefig(
        f"{replica_cross_section_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig2)

fig3, ax3 = plt.subplots(1, figsize = (10, 7))

ax3.scatter(
    phi_array_in_radians, bkm10_bsa_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax3.plot(
    phi_array_in_radians,
    mean_bsa,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_max,
    y2 = bsa_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_90,
    y2 = bsa_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_80,
    y2 = bsa_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_70,
    y2 = bsa_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_60,
    y2 = bsa_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax3.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax3.set_ylabel(r"BSA", fontsize = 16)
ax3.set_title(
    rf"BSA vs. $\phi$, {this_kinematic_set_title_string}"
    "\n"
    f"(KM15): {km15_cff_string}", fontsize = 16)

ax3.legend()
fig3.tight_layout()

replica_bsa_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/dnn_bsa_vs_phi_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig3.savefig(
        f"{replica_bsa_plotname}.{extension}",
        facecolor = 'white')

plt.close(fig3)

cff_h_real_pred_per_replica = np.mean(all_predictions[:, :, 0], axis = 1)
cff_h_imag_pred_per_replica = np.mean(all_predictions[:, :, 1], axis = 1)
cff_ht_real_pred_per_replica = np.mean(all_predictions[:, :, 2], axis = 1)
cff_ht_imag_pred_per_replica = np.mean(all_predictions[:, :, 3], axis = 1)

cff_h_real_mean, cff_h_real_stddev = norm.fit(cff_h_real_pred_per_replica)
cff_h_imag_mean, cff_h_imag_stddev = norm.fit(cff_h_imag_pred_per_replica)
cff_ht_real_mean, cff_ht_real_stddev = norm.fit(cff_ht_real_pred_per_replica)
cff_ht_imag_mean, cff_ht_imag_stddev = norm.fit(cff_ht_imag_pred_per_replica)

print(f"[INFO]: Re[H] mean of {cff_h_real_mean} and stdddev of {cff_h_real_stddev}")
print(f"[INFO]: Im[H] mean of {cff_h_imag_mean} and stdddev of {cff_h_imag_stddev}")
print(f"[INFO]: Re[Ht] mean of {cff_ht_real_mean} and stdddev of {cff_ht_real_stddev}")
print(f"[INFO]: Im[Ht] mean of {cff_ht_imag_mean} and stdddev of {cff_ht_imag_stddev}")

burner_x_values_cff_h_real = np.linspace(
    cff_h_real_mean - 4.*cff_h_real_stddev,
    cff_h_real_mean + 4.*cff_h_real_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_h_real_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_h_real, norm.pdf(burner_x_values_cff_h_real, cff_h_real_mean, cff_h_real_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_real_mean:.3f}$, $\sigma = {cff_h_real_stddev:.3f}$")
ax4.axvline(real_h_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {real_h_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Re$[\mathcal{H}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_h_real_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_h_real_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_h_real_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)

burner_x_values_cff_h_imag = np.linspace(
    cff_h_imag_mean - 4.*cff_h_imag_stddev,
    cff_h_imag_mean + 4.*cff_h_imag_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_h_imag_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_h_imag, norm.pdf(burner_x_values_cff_h_imag, cff_h_imag_mean, cff_h_imag_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_imag_mean:.3f}$, $\sigma = {cff_h_imag_stddev:.3f}$")
ax4.axvline(imag_h_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {imag_h_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Im$[\mathcal{H}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_h_imag_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_h_imag_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_h_imag_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)

burner_x_values_cff_ht_real = np.linspace(
    cff_ht_real_mean - 4.*cff_ht_real_stddev,
    cff_ht_real_mean + 4.*cff_ht_real_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_ht_real_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_ht_real, norm.pdf(burner_x_values_cff_ht_real, cff_ht_real_mean, cff_ht_real_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_ht_real_mean:.3f}$, $\sigma = {cff_ht_real_stddev:.3f}$")
ax4.axvline(real_ht_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {real_ht_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Real$[\tilde{\mathcal{H}}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_ht_real_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_ht_real_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_ht_real_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)

burner_x_values_cff_ht_imag = np.linspace(
    cff_ht_imag_mean - 4.*cff_ht_imag_stddev,
    cff_ht_imag_mean + 4.*cff_ht_imag_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (10, 7))

ax4.hist(cff_ht_imag_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_ht_imag, norm.pdf(burner_x_values_cff_ht_imag, cff_ht_imag_mean, cff_ht_imag_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_ht_imag_mean:.3f}$, $\sigma = {cff_ht_imag_stddev:.3f}$")
ax4.axvline(imag_ht_values[0], color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {imag_ht_values[0]:.3f}")

ax4.set_ylabel("Frequency", rotation = 90., fontsize = 16.)
ax4.set_xlabel(r"Im$[\tilde{\mathcal{H}}]$", fontsize = 16.)
ax4.set_title(f"{this_kinematic_set_title_string}\n(KM15): {km15_cff_string}", fontsize = 16.)

ax4.legend()

cff_ht_imag_plotname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/cff_ht_imag_fits_v{MAJOR_MINOR_NUMBER}"

for extension in ['png', 'eps']:
    fig4.savefig(
        f"{cff_ht_imag_plotname}.{extension}",
        facecolor = 'white')
    
plt.close(fig4)