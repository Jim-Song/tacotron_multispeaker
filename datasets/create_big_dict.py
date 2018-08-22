import json, re

file_chars = 'normal_chars.txt'
file_alpha = 'normal_alphabet.txt'
file_phone = 'normal_phone.txt'
file_punc = 'normal_punctuation.txt'
file_output = 'normal.json'

def get_all_symbols(filename):
    symbols = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                symbols.append(line.strip())
    symbols.append(' ')
    return symbols

symbols = []
symbols += get_all_symbols(file_chars)
symbols += get_all_symbols(file_alpha)
symbols += get_all_symbols(file_phone)
symbols += get_all_symbols(file_punc)

with open(file_output, 'w') as f:
    json.dump(symbols, f)













