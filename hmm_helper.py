from hmm import HMM
import numpy as np
import pickle

SONNET_LINES = 14
NUM_QUATRAINS = 3
QUATRAIN_LINES = 4
COUPLET_LINES = 2
NUM_SHKSP_SONNETS = 151

PUNCTUATION = [',', ':', '.', ';', '?', '!', '(', ')']

SYLL_DICT = 'syll_dict.p'
STRESS_DICT = 'stress_dict.p'

# ----------------------------------------------------------------------
# Read in raw sonnets.
# ----------------------------------------------------------------------

def shksp_raw(filename='shakespeare.txt'):
    seqs = np.loadtxt(filename, delimiter='\n', dtype='str')
    return seqs

# ----------------------------------------------------------------------
# Different tokenizers per line.
# ----------------------------------------------------------------------

# simple_token1:
#   Checks back of each line for punctuation and if found, 
#   replaces with a single token of the punctuation and a 
#   newline character. Punctuation within line is attatched
#   to word on left.
def simple_token1(line):
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    # Handle punctuation at end of line.
    last_word = list(line[-1])
    del line[-1]
    if last_word[-1] in PUNCTUATION:
        tmp = last_word[-1]
        del last_word[-1]
        line.append(''.join(last_word))
        line.append(tmp + '\n')
    else:
        line.append(''.join(last_word))
    return line

# simple_token2:
#   All punctuation are attached to word on left, no 
#   newline characters.
def simple_token2(line):
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line

# simple_token3:
#   All punctuation attached to word on left. Newline
#   characters for each line.
def simple_token3(line):
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line + ['\n']

# simple_token4:
#   Remove all punctuation, no newline character.
def simple_token4(line):
    line = line.lower().lstrip().rstrip()
    for punc in PUNCTUATION:
        line = line.replace(punc, '')
    line = line.split(' ')
    return line

# ----------------------------------------------------------------------
# Preprocess Shakespeare sonnets.
# ----------------------------------------------------------------------

# Produces a single sequence for each sonnet.
def shksp_per_sonnet(tokenizer, filename='shakespeare.txt'):
    raw = shksp_raw(filename)
    sequences = []
    cursor = 0
    for sonnet in range(NUM_SHKSP_SONNETS):
        # Skip first line which is a number.
        cursor += 1
        # Setup sequence.
        seq = []
        for i in range(SONNET_LINES):
            seq += tokenizer(raw[cursor])
            cursor += 1
        sequences.append(seq)
    return sequences

# Produces a single sequence for each line of each sonnet.
def shksp_per_line(tokenizer, filename='shakespeare.txt'):
    raw = shksp_raw(filename)
    sequences = []
    cursor = 0
    for sonnet in range(NUM_SHKSP_SONNETS):
        # Skip first line which is a number.
        cursor += 1
        for i in range(SONNET_LINES):
            sequences.append(tokenizer(raw[cursor]))
            cursor += 1
    return sequences

# Produces two outputs, one with each quatrain (4 lines) as a sequence, and 
# one with each couplet (2 lines) as a sequence.
def shksp_quatrains_and_couplets(tokenizer, filename='shakespeare.txt'):
    raw = shksp_raw(filename)
    quatrains = []
    couplets = []
    cursor = 0
    for sonnet in range(NUM_SHKSP_SONNETS):
        cursor += 1
        couplet = []
        for quatrain in range(NUM_QUATRAINS):
            quatrain = []
            for line in range(QUATRAIN_LINES):
                quatrain += tokenizer(raw[cursor])
                cursor += 1
            quatrains.append(quatrain)
        for line in range(COUPLET_LINES):
            couplet += tokenizer(raw[cursor])
            cursor += 1
        couplets.append(couplet)
    return quatrains, couplets

# Produces two outputs, one with each line of each quatrain as a sequence,
# and one with each line of each couplet as a sequence.
def shksp_quatrain_couplets_line(tokenizer, filename='shakespeare.txt'):
    raw = shksp_raw(filename)
    quatrains = []
    couplets = []
    cursor = 0
    for sonnet in range(NUM_SHKSP_SONNETS):
        cursor += 1
        couplet = []
        for quatrain in range(NUM_QUATRAINS):
            for line in range(QUATRAIN_LINES):
                quatrains.append(tokenizer(raw[cursor]))
                cursor += 1
        for line in range(COUPLET_LINES):
            couplets.append(tokenizer(raw[cursor]))
            cursor += 1
    return quatrains, couplets
    
# ----------------------------------------------------------------------
# Generate from trained models.
# ----------------------------------------------------------------------
# Length is number of syllables to generate. (Generally 10)
def gen_txt(trans, emiss, init, word_map, 
            syll_dict, stress_dict, length=10, 
            space_symb=' '):
    # Verify that the model is functional and setup.
    num_states = len(trans)
    num_words = len(emiss[0])
    assert (num_states == len(trans[0])), 'Transition matrix is not square.'
    assert (num_states == len(emiss)), 'Emission matrix not correct dimensions.'
    
    # Prepare to iterate for words.
    build = ''
    curr_state = np.random.choice(num_states, p=init)
    curr_length = 0
    curr_stress = 0
    
    # Build the sequence.
    while curr_length < length:
        # Select random word based on emission matrix.
        nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
        word = word_map[nxt_token].rstrip('.,?!;:()')
        print word
        
        # Check that word isn't too long and is stressed correctly.
        while (syll_dict[word] + curr_stress > length) or (stress_dict[word] != curr_stress):
                nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
                word = word_map[nxt_token].rstrip('.,?!;:()').lstrip('(')
                print word
        
        build += word_map[nxt_token] + space_symb
        curr_length += syll_dict[word]
        curr_stress = (curr_stress + syll_dict[word]) % 2
        
        # Go to next state.
        curr_state = np.random.choice(num_states, p=trans[int(curr_state)])
    return build
    
# ----------------------------------------------------------------------
# Save/Load trained models.
#   Note that by convention, saved models with have the following
#   suffixes:
#     '-transition.p'
#     '-emission.p'
#     '-wordmap.p'
#     '-init.p'
# ----------------------------------------------------------------------
def load_model(model_name):
    # Load transition matrix.
    save_file = open('matrices/' + model_name + '-transition.p', 'rb')
    trans = pickle.load(save_file)
    save_file.close()
    
    # Load emission matrix.
    save_file = open('matrices/' + model_name + '-emission.p', 'rb')
    emiss = pickle.load(save_file)
    save_file.close()
    
    # Load word map.
    save_file = open('matrices/' + model_name + '-wordmap.p', 'rb')
    wordmap = pickle.load(save_file)
    save_file.close()
    
    # Load initial probability vector.
    save_file = open('matrices/' + model_name + '-init.p', 'rb')
    init = pickle.load(save_file)
    save_file.close()
    
    return trans, emiss, wordmap, init
    
def save_model(model_name, trans, emiss, wordmap, init):
    save_file = open('matrices/' + model_name + '-transition.p', 'wb')
    pickle.dump(trans, save_file)
    save_file.close()

    save_file = open('matrices/' + model_name + '-emission.p', 'wb')
    pickle.dump(emiss, save_file)
    save_file.close()

    save_file = open('matrices/' + model_name + '-wordmap.p', 'wb')
    pickle.dump(wordmap, save_file)
    save_file.close()

    save_file = open('matrices/' + model_name + '-init.p', 'wb')
    pickle.dump(init, save_file)
    save_file.close()
    
# Load syllable and stress dictionaries.
def load_syll_stress_dicts():
    syll_file = open(SYLL_DICT, 'rb')
    syll_dict = pickle.load(syll_file)
    syll_file.close()
    
    stress_file = open(STRESS_DICT, 'rb')
    stress_dict = pickle.load(stress_file)
    stress_file.close()
    
    return syll_dict, stress_dict
    
# ----------------------------------------------------------------------
# Train and save models.
# ----------------------------------------------------------------------
def train_model(model_name, num_states, data):
    hmm = HMM(num_states)
    wordmap, init, trans, emiss = hmm.train(data, epsilon=0.01)
    save_model(model_name, trans, emiss, wordmap, init)
    