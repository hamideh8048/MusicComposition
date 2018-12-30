import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from keras.layers import LSTM
from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.utils import np_utils
import numpy as np
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def get_notes():
    ####################################نت ها از فایل midi  خوانده می شوند #################################
    notes = []

    for file in glob.glob("Pokemon MIDIs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def prepare_sequences(notes, n_vocab):
    ####################################ساخت دنباله برای ورودی و خروجی مدل #################################
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    ####################################دیکشنری برای map کردن گام نت به int #################################
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    ####################################تولید دنباله ورودی .به ازای هر دنباله ورودی یک نت خروجی داریم که در یک آرایه ذخیره میکنیم #################################
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)


    ####################################تغییر ورودی به فرمتی که برای lstm قابل قبول باشد #################################
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    ####################################نرمال کردن ورودی بین -1 و1 #################################
    network_input = (network_input - float(n_vocab) / 2) / (float(n_vocab) / 2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)




def create_midi(prediction_output, filename):
    ####################################تبدیل دنباله خروجی به نت وسپس تولید فایل موسیقی#################################
    offset = 0
    output_notes = []


    for item in prediction_output:
        pattern = item[0]
        ####################################در اینجا کتابخانه  note21 تشخیص میدهد که  آکورد هست یا نت معمولی#################################

        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output8.mid')



randomDim = 1000

####################################خواندن داده(نت)#################################
notes = get_notes()
n_vocab = len(set(notes))
X_train, y_train = prepare_sequences(notes, n_vocab)

seq_length = 100
seq_shape = (seq_length, 1)
# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

disc_loss = []
gen_loss = []

####################################شبکه اول Generator  است که یک آرایه به صورت رندم به عنوان ورودی میگیرد  این شبکه سعی می کند دنباله ای از نت را مشابه آنچه در مجموعه آموزش  است به خروجی بدهد#################################
generator = Sequential()
generator.add(Dense(256, input_dim=randomDim))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(np.prod(seq_shape), activation='tanh'))
generator.add(Reshape(seq_shape))
generator.compile(loss='binary_crossentropy', optimizer=adam)

####################################شبکه دوم Discriminator است که یک شبکه binary classification  است که آموزش داده شده تا دنباله تولید شده توسط generator را ارزیابی کند. این شبکه دیتا واقعی را از مجموعه آموزش میگیرد همچنین  ورودی دیگری  از شبکه generator  میگیرد و وظیفه آن تشخیص دنباله نت واقعی از غیر واقعی(نویز) است#################################

discriminator = Sequential()
discriminator.add(LSTM(512, input_shape=seq_shape, return_sequences=True))
discriminator.add(Bidirectional(LSTM(512)))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

####################################هاین شبکه دیتا واقعی را از مجموعه آموزش میگیرد و مچنین  ورودی دیگری  از شبکه generator  میگیرد و وظیفه آن تشخیص دنباله نت واقعی از غیر واقعی(نویز) است#################################

discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []


def generate( input_notes):

    notes = input_notes
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    #################################### به صورت رندم نویز تولید کرده و به عنوان ورودی به generatorld می دهیم#################################

    noise = np.random.normal(0, 1, (1, 1000))
    predictions = generator.predict(noise)

    pred_notes = [x * 242 + 242 for x in predictions[0]]

    pred_notes = [int_to_note[int(x)] for x in pred_notes]

    create_midi(pred_notes, 'gan_final')

def train(epochs, batchSize):


    notes = get_notes()
    n_vocab = len(set(notes))
    X_train, y_train = prepare_sequences(notes, n_vocab)
    batchCount = X_train.shape[0] / batchSize
    print
    'Epochs:', epochs
    print
    'Batch size:', batchSize
    print
    'Batches per epoch:', batchCount
    real = np.ones((batchSize, 1))
    fake = np.zeros((batchSize, 1))
    for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batchSize)
            real_seqs = X_train[idx]
            noise = np.random.normal(0, 1, (batchSize, randomDim))
            # Generate fake sequence
            gen_seqs = generator.predict(noise)



            # Train discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = discriminator.train_on_batch(gen_seqs, fake)
            #discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            discriminator.trainable = False
            #generator loss
            g_loss = gan.train_on_batch(noise, real)
            #check again Hamide

            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
            disc_loss.append(d_loss)
            gen_loss.append(g_loss)
    generate(notes)



if __name__ == '__main__':
    train(10, 32)