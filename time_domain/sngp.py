import pickle, random
import tensorflow as tf
from tensorflow.keras import layers, models
X_train, Y_train, X_test, Y_test = [], [], [], []
print("Reached 1")
all_data = pickle.load(open("kmci_kctrl_kdem_14_15_notevt_200.pkl", 'rb'))
random.shuffle(all_data['kctrl'])
random.shuffle(all_data['kmci'])
print("Reached 1")
pickle.dump(all_data['kctrl'], open("kctrl_shuffled.pkl", "wb"))
pickle.dump(all_data['kmci'], open("kmci_shuffled.pkl", "wb"))
print("Reached 1")

train_ind_to_patient = {}
test_ind_to_patient = {}

print(all_data.keys())
print("Reached 1")
l_ctrl = len(all_data['kmci'])
for i in range(int(0.75*l_ctrl)):
  ith = all_data['kmci'][i]
  for vec in ith:
    X_train.append(vec)
    Y_train.append(1)
    train_ind_to_patient[len(X_train)-1] = (i, 'kmci')

print("Reached 1")
for i in range(int(0.75*l_ctrl)+1, l_ctrl):
  ith = all_data['kmci'][i]
  for vec in ith:
    X_test.append(vec)
    Y_test.append(1)
    test_ind_to_patient[len(X_test)-1] = (i, 'kmci')

print("Reached 1")
l_ctrl = len(all_data['kctrl'])
for i in range(int(0.75*l_ctrl)):
  ith = all_data['kctrl'][i]
  for vec in ith:
    X_train.append(vec)
    Y_train.append(0)
    train_ind_to_patient[len(X_train)-1] = (i, 'kctrl')

print("Reached 1")
for i in range(int(0.75*l_ctrl)+1, l_ctrl):
  ith = all_data['kctrl'][i]
  for vec in ith:
    X_test.append(vec)
    Y_test.append(0)
    test_ind_to_patient[len(X_test)-1] = (i, 'kctrl')

print("Reached 1")
pickle.dump(test_ind_to_patient, open("test_ind_to_patient.pkl", "wb"))
pickle.dump(train_ind_to_patient, open("train_ind_to_patient.pkl", "wb"))

print("Reached 1")
print(len(X_train), len(Y_train), len(X_test), len(Y_test))
import numpy
X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)
X_test = numpy.array(X_test)
Y_test = numpy.array(Y_test)
print(X_train.shape, Y_train.shape, X_test.shape)

import tensorflow as tf
import official.nlp.modeling.layers as nlp_layers
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def create_1d_cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(filters=56, kernel_size=40, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=32, kernel_size=15, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=20, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)


    dense_layer = layers.Dense(620, activation=None)
    x = nlp_layers.SpectralNormalization(dense_layer, norm_multiplier=0.9)(x)
    x = layers.ReLU()(x)

    dense_layer = layers.Dense(200, activation=None)
    x = nlp_layers.SpectralNormalization(dense_layer, norm_multiplier=0.9)(x)
    x = layers.ReLU()(x)


    gp_layer = nlp_layers.RandomFeatureGaussianProcess(units=2, gp_cov_momentum=0.9)
    outputs = gp_layer(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.gp_layer = gp_layer

    return model

input_shape = (200, 17)
model = create_1d_cnn_model(input_shape)

#model.compile(optimizer=tf.keras.optimizers.Adam(),
#              loss=tf.keras.losses.BinaryCrossentropy(),
#              metrics=[tf.keras.metrics.BinaryAccuracy()])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=[tf.keras.metrics.BinaryAccuracy()])

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[metrics])
model.summary()

class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name

    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the beginning of the epoch."""
        if epoch > 0:
            layer = self.model.get_layer(self.layer_name)
            if hasattr(layer, 'reset_covariance_matrix'):
                layer.reset_covariance_matrix()
reset_cov_callback = ResetCovarianceCallback(layer_name=model.gp_layer.name)

es = EarlyStopping(monitor='val_random_feature_gaussian_process_binary_accuracy', mode='max', verbose=1, patience=2)
model.fit(np.transpose(X_train, (0, 2, 1)), Y_train, epochs=10, batch_size=32, validation_data=(np.transpose(X_test, (0, 2, 1)), Y_test), callbacks=[reset_cov_callback, es])
def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.):
  # Computes uncertainty-adjusted logits using the built-in method.
  logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(
      logits, covmat, mean_field_factor=lambda_param)

  return tf.nn.softmax(logits_adjusted, axis=-1)[:, 0]

print(model.evaluate(np.transpose(X_test, (0, 2, 1)), Y_test, batch_size=32))
sngp_logits, sngp_covmat = model(np.transpose(X_test, (0, 2, 1)), training=False)
print(sngp_logits)
corr = 0
tot = 0
corr_probs = []
wrong_probs = []
for logit, lab in zip(sngp_logits, Y_test):

    soft = np.exp(logit)/sum(np.exp(logit))
    pred = 0
    good = False
    if logit[1] >= logit[0]:
        pred = 1
    if pred == 0 and lab == 0:
        corr += 1
        good = True
    if pred == 1 and lab == 1:
        corr += 1
        good = True

    if good:
        corr_probs.append(soft[lab])
    else:
        wrong_probs.append(soft[pred])

    tot += 1
print(corr, tot)
pickle.dump(corr_probs, open("corr_probs.pkl", "wb"))
pickle.dump(wrong_probs, open("wrong_probs.pkl", "wb"))
# sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]
# print("Diag part done", sngp_variance)
# sngp_logits_adjusted = sngp_logits / tf.sqrt(1. + (np.pi / 8.) * sngp_variance)
# print("Logits adj", sngp_logits_adjusted)
# sngp_probs = tf.nn.softmax(sngp_logits_adjusted, axis=-1)[:, 0]
# print(sngp_probs)

# sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
# print(sngp_probs)

