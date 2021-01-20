import pickle
from nltk import word_tokenize, ngrams

def part2():
    # read in pickled dicts
    eng1, eng2 = pickle.load(open('english.pickle', 'rb'))
    fre1, fre2 = pickle.load(open('french.pickle', 'rb'))
    ital1, ital2 = pickle.load(open('italian.pickle', 'rb'))

    # open test file and gets lines
    test_file = open('training_files/LangId.test')
    test_text = test_file.read()
    tokens = test_text.splitlines()

    # output file
    out_file = open('output.txt', 'w')

    # calculate total vocab length
    v = len(eng1) + len(fre1) + len(ital1)

    counter = 0
    for line in tokens:
        # call computing function for each language
        eng_p = compute(line, eng1, eng2, v)
        fre_p = compute(line, fre1, fre2, v)
        ital_p = compute(line, ital1, ital2, v)

        # check which probability is largest
        if (eng_p > fre_p) and (eng_p > ital_p):
            lang = "English"
        elif (fre_p > eng_p) and (fre_p > ital_p):
            lang = "French"
        else:
            lang = "Italian"

        # write to output file
        counter = counter + 1
        out_file.write(str(counter))
        out_file.write(" ")
        out_file.write(lang)
        out_file.write("\n")
    # close output file so we can reopen with 'r'
    out_file.close()

    count1 = 0
    count2 = 0
    counter = 0
    print("Line numbers of incorrectly classified lines: ")
    # calculate accuracy and print wrong line numbers
    with open('training_files/LangId.sol', 'rU') as f1, open('output.txt', 'rU') as f2:
        for line1, line2 in zip(f1, f2):
            counter = counter + 1
            if line1 != line2:
                print(counter)
                count1 = count1 + 1
            else:
                count2 = count2 + 1

    # print accuracy
    print("Accuracy: ", (count2 / (count1 + count2)) * 100, "%")

    # close files
    test_file.close()
    out_file.close()


# function for calculating probability
def compute(line, unigram_dict, bigram_dict, V):
    unigrams_test = word_tokenize(line)
    bigrams_test = list(ngrams(unigrams_test, 2))

    p = 1

    # calculate probability using laplace smoothing
    for bigram in bigrams_test:
        b = bigram_dict[bigram] if bigram in bigram_dict else 0
        u = unigram_dict[bigram[0]] if bigram[0] in unigram_dict else 0
        p = p * ((b + 1) / (u + V))
    return p