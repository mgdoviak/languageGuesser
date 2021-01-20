import pickle
import language_guesser2
from nltk import word_tokenize, ngrams

def main(file1):
 # open and read file
 file = open(file1, 'r', encoding = 'utf-8')
 raw_text = file.read()

 # remove new lines
 raw_text = raw_text.replace('\r', '').replace('\n', '')

 # tokenize
 tokens = word_tokenize(raw_text)

 # get bigrams list
 bigrams = list(ngrams(tokens, 2))

 # get unigrams list
 unigrams = tokens

 # create bigram dictionary
 bigram_dict = {b:bigrams.count(b) for b in set(bigrams)}

 # create unigram dictionary
 unigram_dict = {t:unigrams.count(t) for t in set(unigrams)}

 return unigram_dict, bigram_dict


if __name__ == '__main__':

 # call function 3 times for the 3 files
 eng_dict1, eng_dict2 = main('training_files/LangId.train.English')
 fre_dict1, fre_dict2 = main('training_files/LangId.train.French')
 ital_dict1, ital_dict2 = main('training_files/LangId.train.Italian')

 # pickle the dicts
 two_eng_dict = [eng_dict1, eng_dict2]
 two_fre_dict = [fre_dict1, fre_dict2]
 two_ital_dict = [ital_dict1, ital_dict2]

 pickle.dump(two_eng_dict, open('english.pickle', 'wb'))
 pickle.dump(two_fre_dict, open('french.pickle', 'wb'))
 pickle.dump(two_ital_dict, open('italian.pickle', 'wb'))

# call second .py file
 language_guesser2.part2()