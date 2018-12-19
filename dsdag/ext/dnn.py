from dsdag.core.op import OpVertex
from dsdag.core.parameter import BaseParameter, DatetimeParameter, UnhashableParameter

from keras.models import Model
from keras.optimizers import *
from keras.layers import *

class DenseAutoencoder(OpVertex):
    input_feature_dim = BaseParameter()
    output_feature_dim = BaseParameter()
    encoding_dim = BaseParameter(25)
    dropout = BaseParameter(.15)
    input_dropout = BaseParameter(0.15)
    width = BaseParameter(128)
    depth = BaseParameter(1)
    hidden_activation = BaseParameter('relu')
    output_activation = BaseParameter('relu')
    lr = BaseParameter(0.001)
    decay = BaseParameter(0.0)
    optimizer = BaseParameter(None)
    loss = BaseParameter('mean_squared_error')
    batchnorm = BaseParameter(True)
    kernel_initializer = BaseParameter('glorot_normal')
    bias_initializer = BaseParameter('zeros')

    @staticmethod
    def build_autoencoder(input_feature_dim,
                          output_feature_dim,
                          encoding_dim=25,
                          dropout=.15,
                          input_dropout=0.15,
                          width=128,
                          depth=1,
                          hidden_activation='relu',
                          output_activation='relu',
                          lr=0.001, decay=0.0,
                          optimizer=None,
                          loss='mean_squared_error',
                          batchnorm=True,
                          kernel_initializer='glorot_normal',
                          bias_initializer='zeros'):
        """
        returns encoder, decoder, autoencoder
        """

        #print("Input-Enc-Output: %d-%d-%d" % (input_feature_dim, encoding_dim, output_feature_dim))
        input_sym = Input(shape=(input_feature_dim,))
        enc_x = input_sym

        if input_dropout is not None and input_dropout > 0.:
            enc_x = Dropout(input_dropout)(enc_x)

        for d in range(depth):
            enc_x = Dense(width, activation=None,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)(enc_x)
            if batchnorm:
                enc_x = BatchNormalization()(enc_x)
            enc_x = Activation(hidden_activation)(enc_x)
            enc_x = Dropout(dropout)(enc_x)

        encoded = Dense(encoding_dim, activation=None,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)(enc_x)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
        encoded = Activation(hidden_activation, name='encoder')(encoded)

        decoder = Dropout(dropout)(encoded)
        dec_x = decoder

        for d in range(depth):
            dec_x = Dense(width, activation=None, name='decoder_l%d' % d,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)(dec_x)
            if batchnorm:
                dec_x = BatchNormalization()(dec_x)
            dec_x = Activation(hidden_activation)(dec_x)
            dec_x = Dropout(dropout)(dec_x)

        decoded = Dense(output_feature_dim, activation=None,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer)(dec_x)
        decoded = Activation(output_activation)(decoded)

        autoencoder = Model(input_sym, decoded)

        encoder = Model(input_sym, encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        # decoder_layer = autoencoder.layers[-(depth*4 + 1)]
        decoder_layer = [l for l in autoencoder.layers if l.name == 'decoder_l0'][0]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        if optimizer is None:
            optimizer = Adam(lr=lr, decay=decay)

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return encoder, decoder, autoencoder