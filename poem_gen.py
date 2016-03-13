import hmm_helper as hh

# Method: gpoem_qac_models
#     Generates Shakespearean sonnets that follow the meter and syllable 
#     counts for each line. Punctuation is also variable under these models. 
#     Takes in the models trained on entire quatrains and entire couplets. 
#     The poem generated does not necessarily rhyme due to difficulties with 
#     seeding for forward models. The valid models for this poem generation 
#     are as follows:
#         1. 20-states-full
#         2. 30-states-full
def gpoem_qac_models(model_name):
    q_trans, q_emiss, q_wm, q_init = hh.load_model(model_name + '-quatrains')
    c_trans, c_emiss, c_wm, c_init = hh.load_model(model_name + '-couplets')
    poem = hh.gen_poem(q_trans, q_emiss, q_init, q_wm,
                       c_trans, c_emiss, c_init, c_wm)
    return poem

# Method: gpoem_rev_models
#     Generates Shakespearean sonnets that follow the correct number of
#     syllables, rhymes, and has reasonable punctuation. The poems, however
#     do not necessarily follow the iambic pentameter stressing. Takes in 
#     reversed models trained on a per-line sequencing. The valid models for
#     this poem generation are as follows:
#         1. 5-state-rev-lines
#         2. 8-state-rev-lines
#         3. 10-state-rev-lines
#         4. 20-state-rev-lines
#         5. 30-state-rev-lines
#         6. 50-state-rev-lines
#         7. 75-state-rev-lines
#         8. 100-state-rev-lines
def gpoem_rev_models(model_name):
    trans, emiss, wm, init = hh.load_model(model_name)
    inv_wm = {v: k for k, v in wm.items()}
    poem = hh.gen_poem_rhyme1(trans, emiss, wm, inv_wm)
    return poem