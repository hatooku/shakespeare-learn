HMM, Poem Generation, and Visualization Instructions
----------------------------------------------------

Main Files:
    hmm.py (Baum-Welch algorithm)
    hmm_helper.py (Preprocessing, Tokenization, Saving Matrices)
    train.py (Off-the-shelf algorithm example)
    data_vis.py (Data Visualization)
    poem_gen.py (Poem Generation)

HMM Algorithm:
    from hmm.py import HMM
    # EXAMPLE
    h = HMM(2)
    data = [['R', 'W', 'B', 'B']] # One sequence
    h.train(data)
    print h.A
    print h.O
    print h.PI

Train Model and Save:
    from hmm_helper import *
    train_model("Model_name", 20, data)

Visualization:



Poem Generation: