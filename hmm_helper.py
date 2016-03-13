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
    
def gen_rhyme_lists(filename='shakespeare.txt'):
    raw = shksp_raw(filename)
    quatrains_rhymes = []
    couplets_rhymes = []
    cursor = 0
    for sonnet in range(NUM_SHKSP_SONNETS):
        cursor += 1
        for quatrain in range(NUM_QUATRAINS):
            line0 = simple_token2(raw[cursor])
            line1 = simple_token2(raw[cursor + 1])
            line2 = simple_token2(raw[cursor + 2])
            line3 = simple_token2(raw[cursor + 3])
            quatrains_rhymes.append((line0[-1], line2[-1]))
            quatrains_rhymes.append((line1[-1], line3[-1]))
            cursor += 4
        line0 = simple_token2(raw[cursor])
        line1 = simple_token2(raw[cursor + 1])
        couplets_rhymes.append((line0[-1], line1[-1]))
        cursor += 2
    return quatrains_rhymes, couplets_rhymes
    
# ----------------------------------------------------------------------
# Generate from trained models.
# ----------------------------------------------------------------------
# Length is number of syllables to generate. (Generally 10)
# Generates a line using a init vector to choose start state.
def gen_txt(trans, emiss, init, word_map, 
            syll_dict, stress_dict, length=10, 
            space_symb=' ', curr_state=None):
    # Verify that the model is functional and setup.
    num_states = len(trans)
    num_words = len(emiss[0])
    assert (num_states == len(trans[0])), 'Transition matrix is not square.'
    assert (num_states == len(emiss)), 'Emission matrix not correct dimensions.'
    
    # Prepare to iterate for words.
    build = ''
    if curr_state is None:
        curr_state = np.random.choice(num_states, p=init)
    curr_length = 0
    curr_stress = 0
    
    # Build the sequence.
    while curr_length < length:
        # Select random word based on emission matrix.
        nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
        word = word_map[nxt_token].rstrip('.,?!;:()').lstrip('(')
        # print word
        
        # Check that word isn't too long and is stressed correctly.
        while (syll_dict[word] + curr_stress > length) or (stress_dict[word] != curr_stress):
                nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
                word = word_map[nxt_token].rstrip('.,?!;:()').lstrip('(')
        
        build += word_map[nxt_token] + space_symb
        curr_length += syll_dict[word]
        curr_stress = (curr_stress + syll_dict[word]) % 2
        
        # Go to next state.
        curr_state = np.random.choice(num_states, p=trans[int(curr_state)])
    return build, curr_state

def count_syllables_stress(line, syll_dict, stress_dict):
    count = 0
    stress = True
    curr_stress = 0
    words = simple_token2(line)
    for word in words:
        word = word.rstrip('.,?!;:()').lstrip('(')
        count += syll_dict[word]
        if stress_dict[word] != curr_stress:
            stress = False
        curr_stress = (curr_stress + syll_dict[word]) % 2
    return count, stress
    
# Generates a line with a seeded last word. (Omit stress consideration.)
# Requires both maps, int->word and word->int.
def gen_txt_rhyme1(trans, emiss, wti_map, itw_map, syll_dict, stress_dict,
                  seed, length=10):
    # Verify that the model is functional and setup.
    num_states = len(trans)
    num_words = len(emiss[0])
    assert (num_states == len(trans[0])), 'Transition matrix is not square.'
    assert (num_states == len(emiss)), 'Emission matrix not correct dimensions.'
    
    # Find most likely state to emit seed.
    token_num = wti_map[seed]
    best_state = np.argmax(emiss[:, token_num])
    build = seed
    # Transpose transition matrix since we are building line backwards.
    ttrans = np.transpose(trans)
    while count_syllables_stress(build, syll_dict, stress_dict) != (10, True):
        curr_state = best_state
        build = seed
        while count_syllables_stress(build, syll_dict, stress_dict)[0] < 10:
            nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
            word = itw_map[nxt_token]
            build = word + ' ' + build
            # Go to next state.
            curr_state = np.random.choice(num_states, 
                                          p=ttrans[int(curr_state)] / np.sum(ttrans[int(curr_state)]))
    return build
    
# Generates a line with a seeded last word. Assumes that the model was trained
# backwards.
# Requires both maps, int->word and word->int.
def gen_txt_rhyme2(trans, emiss, wti_map, itw_map, syll_dict, stress_dict,
                  seed, length=10):
    # Verify that the model is functional and setup.
    num_states = len(trans)
    num_words = len(emiss[0])
    assert (num_states == len(trans[0])), 'Transition matrix is not square.'
    assert (num_states == len(emiss)), 'Emission matrix not correct dimensions.'
    
    # Find most likely state to emit seed.
    token_num = wti_map[seed]
    best_state = np.argmax(emiss[:, token_num])
    build = seed
    while count_syllables_stress(build, syll_dict, stress_dict)[0] != 10:
        curr_state = best_state
        build = seed
        while count_syllables_stress(build, syll_dict, stress_dict)[0] < 10:
            # Go to next state.
            curr_state = np.random.choice(num_states, p=trans[int(curr_state)])
            nxt_token = int(np.random.choice(num_words, p=emiss[int(curr_state)]))
            word = itw_map[nxt_token]
            build = word + ' ' + build
    return build

# Generates an un-rhymed poem.
def gen_poem(q_trans, q_emiss, q_init, q_wm, 
             c_trans, c_emiss, c_init, c_wm):
    # Set up the generator.
    poem = ''
    syll_dict, stress_dict = load_syll_stress_dicts()
    # Generate the quatrains.
    for q in range(NUM_QUATRAINS):
        q_state = None
        quatrain = ''
        for line in range(QUATRAIN_LINES):
            q_line, q_state = gen_txt(q_trans, q_emiss, q_init, q_wm,
                                      syll_dict, stress_dict, 
                                      curr_state=q_state)
            poem += q_line + '\n'
    # Generate the final couplet.
    c_state = None
    for line in range(COUPLET_LINES):
        c_line, c_state = gen_txt(c_trans, c_emiss, c_init, c_wm,
                                   syll_dict, stress_dict,
                                   curr_state=c_state)
        poem += c_line + '\n'
    return poem

def gen_poem_rhyme(trans, emiss, wm, iwm):
    # Set up generator and important dictionaries.
    poem = ''
    syll_dict, stress_dict = load_syll_stress_dicts()
    q_rhyme, c_rhyme = gen_rhyme_lists()
    # Generate the quatrains.
    for q in range(NUM_QUATRAINS):
        rhymes = np.random.choice(len(q_rhyme), 2)
        end0 = q_rhyme[rhymes[0]][0]
        end1 = q_rhyme[rhymes[1]][0]
        end2 = q_rhyme[rhymes[0]][1]
        end3 = q_rhyme[rhymes[1]][1]
        
        qline0 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                               syll_dict, stress_dict, end0)
        qline1 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                               syll_dict, stress_dict, end1)
        qline2 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                               syll_dict, stress_dict, end2)
        qline3 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                               syll_dict, stress_dict, end3)
        poem += qline0 + '\n' + qline1 + '\n' + qline2 + '\n' + qline3 + '\n'
    # Generate the couplet.
    rhymes = np.random.choice(len(c_rhyme))
    end0 = c_rhyme[rhymes][0]
    end1 = c_rhyme[rhymes][1]
    cline0 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                            syll_dict, stress_dict, end0)
    cline1 = gen_txt_rhyme2(trans, emiss, iwm, wm, 
                            syll_dict, stress_dict, end1)
    poem += cline0 + '\n' + cline1 + '\n'
    return poem
    
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
    