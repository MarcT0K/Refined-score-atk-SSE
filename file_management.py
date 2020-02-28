import struct

from ctypes import c_int, c_double

import numpy as np


def build_cooccurence_bin(coocc_mat: np.array, filename="cooccurrence.bin"):
    """Produces a binary file using the format expected by the Stanford C implementation
    of GloVe ([word_i:int][word_j:int][cocc_val:double]).
    
    Arguments:
        coocc_mat {np.array} -- Co-occurence matrix used to build the binary file.
    
    Keyword Arguments:
        filename {str} -- Name of the binary file (default: {"cooccurence.bin"})
    """
    with open(filename, "wb") as bin_file:
        for coord, coocc_val in np.ndenumerate(coocc_mat):
            if coocc_val:
                byte_array = [coord[0], coord[1], float(coocc_val)]
                bin_file.write(struct.pack("iid", *byte_array))


def build_voc_file(voc_freq_list, filename="vocab.txt"):
    """Creates a file containing line with this format: '[word] [occurence]'.
    It is the format used in the Stanford C implementation of GloVe.
    
    Arguments:
        voc_freq_list {[type]} -- List of tuples ([word], [occurence])
    
    Keyword Arguments:
        filename {str} -- File where the vocabulary will be written (default: {"vocab.txt"})
    """
    with open(filename, "w") as voc_file:
        for word, occ in voc_freq_list:
            voc_file.write(f"{word} {occ}\n")
