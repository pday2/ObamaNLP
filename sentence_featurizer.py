"""
Creates tidy table of variables from text

Usage: python the_featurizer folder source_name date_file.csv
arguments:
   folder: string of folder or directory names
   source_name: string to use for source
   date_file.csv: string of existing file name with dates and full text file names, with directory
returns: nothing
creates: some files with data in them
"""
# The featurizer
# Command line arguments: directory, suffix, dates file
# python sentence_featurizer.py Data amrhet datetitle.csv
# python sentence_featurizer.py Top10 topten datefiles_topten.csv
import pandas as pd
import numpy as np
import spacy
import en_core_web_md
import regex as re
import os
import sys
import unicodedata
import re
import stanza
from collections import defaultdict
import nltk
from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict
from nrclex import NRCLex
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPDependencyParser
stanza.download('en')
nltk.download('punkt')
word_token = TreebankWordTokenizer()
nltk.download ('wordnet')

# For dependency parse tree depth
# Using Stanford's CoreNLP parser with NLTK
# 1. Download CoreNLP from https://stanfordnlp.github.io/CoreNLP/download.html
# 2. make sure Java is installed, otherwise download and install Java - https://www.java.com/en/download/windows_manual.jsp
# 3. Unzip/extract CoreNLP zip file to a directory
# 4. Go to that directory and open a command terminal, and run the following command...
# 4b. on my laptop its in C:\Users\peter\stanford-corenlp-4.5.2
# 5. java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# 6. Now for graphviz if you want to view the parse trees, download from https://graphviz.org/download/ then install
# 7. Now, can run the following python code

def main():
    print('!!! Note: Make sure Stanford CoreNLP server has been started for parse tree depth !!!')
    if len(sys.argv[1]) > 0:
        paths = sys.argv[1]
    else:
        paths = 'Data'
    if '.' not in paths:
        paths = ['./'+paths+'/']
    if len(sys.argv[2]) > 0:
        suffix = sys.argv[2]
    else:
        suffix = 'amrhet'
    source = suffix
    if len(sys.argv[3]) > 0:
        dates_file = sys.argv[3]
    else:
        dates_file = 'datetitle.csv'  
    print('Directory:',paths)
    print('Suffix:',suffix)
    print('Dates File:',dates_file)
    # Initialize English neural pipeline
    stanza.download('en')
    nlp_stanza = stanza.Pipeline('en', processors='tokenize, lemma, pos')
    nlp_spacy = spacy.load("en_core_web_md")
    stopwords = pd.read_table('./word_lists/kaggle_stopwords.txt')
    dates = pd.read_csv(dates_file)
    dates['date'] = pd.to_datetime(dates['date'], format='%Y-%m-%d')
    # date, title, file
    try:
        dates.rename(columns={"url":"file"}, inplace=True)
    except:
        print()
    try:
        dates = dates.drop('title', axis=1)
    except:
        print()
    
    print('---------LOADING DOCUMENTS----------')
    # Load up the speeches
    speeches = []
    for path in paths:
        list_of_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    list_of_files.append(os.path.join(root,file))

        for file in list_of_files:
            with open(file, encoding='utf-8') as f:
                text = f.read()
            f.close()
            speeches.append(text)

    #clean out goofy unicode  space characters 
    speeches = [unicodedata.normalize("NFKD", speech) for speech in speeches if len(speech)>0 ]
    #clean out xa0 space characters
    [speech.replace(u'\xa0', '') for speech in speeches]; # ; supresses output
    # remove [stuff] in between square brackets
    def remove_bracket(text):
        return re.sub(r'(\[[^w]*\]\s)', '',text)
    speeches = [remove_bracket(speech) for speech in speeches]
    # Clean up whitespace
    speeches = [re.sub('[\s+]', ' ', speech) for speech in speeches]
    # Remove -- that's all over the amrhet files
    def remove_dashes(text):
        return re.sub(r'-- ', '', text)
    speeches = [remove_dashes(speech) for speech in speeches]
    text_df = pd.DataFrame({'file' : list_of_files,
                            'text' : speeches})

    text_df = pd.merge(text_df, dates, how='inner', on='file')
    text_df = text_df.sort_values(by='date', ignore_index=True)
    text_df = text_df[['date', 'file', 'text']]
    text_df['source'] = source
    text_df.set_index('date', inplace=True)
    text_df['sentences'] = text_df['text'].apply(sent_tokenize)
    text_df['words'] = text_df['text'].apply(word_token.tokenize)
    text_df['word_set'] = text_df['words'].apply(set)
    text_df['num_sents'] = text_df['sentences'].apply(len)
    text_df['num_words'] = text_df['words'].apply(len)
    text_df['num_unique_words'] = text_df['word_set'].apply(len)
    
    ################# BREAK UP INTO SENTENCES ########################
    # Break up into one row per sentence
    text_df=text_df.explode('sentences')
    text_df.drop(['text', 'words', 'word_set'] , axis=1, inplace=True)
    text_df.rename(columns={"sentences": "text"}, inplace=True)
    
    
    
    print("Length of text_df", len(text_df))
    ############# POS TAGGING ###################  NEW TRY!!!!
    print('---------POS TAGGING---------')

    # The nlp(text) uses a lot of gpu memory, causes errors sometimes, may need to restart notebook to freshen things up
    parts_of_speech = ['NUM','ADV','SYM','NOUN','ADP','PROPN','DET','INTJ','AUX',
                       'CCONJ','ADJ','PRON','SCONJ','X','VERB','PUNCT','PART']
    for col in parts_of_speech:
        text_df[col] = 0
    print("Length of text_df:", len(text_df))
    print('...pst, this is slow')
    for i, text in enumerate(text_df.text):
        doc = nlp_stanza(text) # Run stanza on each speech
        mat_of_pos = [[word.pos for word in sentence.words] for sentence in doc.sentences] # matrix of POS for each sentence
        # How to flatten a list = [item for sublist in list_of_lists for item in sublist]
        list_of_pos = [pos for sentence in mat_of_pos for pos in sentence] # flatten matrix into one list of all pos
        total_pos_count = len(list_of_pos)
        for pos in parts_of_speech:
            #dfd.iloc[[0, 2], dfd.columns.get_loc('A')]
            text_df.iloc[i, text_df.columns.get_loc(pos)] = list_of_pos.count(pos)/total_pos_count
            
    print("Length of text_df", len(text_df))
    #################### NRCLex EMOTIONS ########################
    print('---------NRCLex EMOTION TAGGING---------')
    text_df['emo'] = text_df.text.apply(NRCLex)
    text_df['emo_list'] = {}

    for i in range(len(text_df)):
        anticip = text_df.emo[i].affect_frequencies.pop('anticip')
    print(text_df.shape)

    # Get names of emotion attributes, locate and remove anticipation as it seems to alway be 0
    attributes = ['fear','anger','trust','surprise','positive','negative',
                'sadness','disgust','joy','anticipation']
    text_df = text_df.reindex(columns=text_df.columns.tolist() + attributes)

    for i in range(len(text_df)):
        for attr in attributes:
            try:
                value = text_df.emo[i].affect_frequencies[attr]
            except:
                value = 0
            text_df.iloc[i, text_df.columns.get_loc(attr)] = value
    
    print("Length of text_df", len(text_df))
    ##################### TEXT BLOB ################################
    print('---------TextBlob----------')
    text_df['TBsubjectivity']=[TextBlob(text).sentiment.subjectivity for text in text_df['text']]
    text_df['TBpolarity']=[TextBlob(text).sentiment.polarity for text in text_df['text']]
    print("Length of text_df", len(text_df))
    ########################### READABILITY ################################
    print('---------READABILITY----------')
    
    ########## HELPER FUNCTIONS #############
    def words_per_sentence(sentence):
        '''returns: integer number of words in a sentence'''
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        return(len(tokens))

    def chars_per_word(word):
        '''returns: integer number of characters in a word'''
        return(len(word))

    def string_to_list(sentence):
        '''converts a string/sentence to a list of words'''
        tokenizer = RegexpTokenizer(r'\w+')
        return(tokenizer.tokenize(sentence))

    def chars_per_word_sentence(sentence):
        '''input: string of a sentence
           returns: list of number of characters in a sentence'''
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        char_len_list = [chars_per_word(word) for word in tokens]
        return(char_len_list)

    def text_to_sentence(text):
        '''uses spacy nlp object to break up sentences
           input: pandas series of strings
           returns: list of sentence strings'''
        doc = nlp(' '.join(text.tolist()))
        assert doc.has_annotation("SENT_START")
        return([str(sent) for sent in doc.sents])

    def text_to_wordlist(text):
        '''input: string or pandas series of text
           returns: list of all words'''
        if isinstance(text, str):
            text = [text]
        return(' '.join(text).split())

    def syllable_count(word):
        '''counts number of syllables in a word'''
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    def count_total_words(text):
        '''count total number of words in a text
           input: str or pandas.core.series.Series of text
           returns: integer'''
        if isinstance(text, list):
            list_of_sentence = text
        elif isinstance(text, str):
            list_of_sentence = [text]
        elif isinstance(text, pd.Series):
            list_of_sentence = text_to_sentence(text)
        else:
            print('count_total_words: Error: not a string or pandas series object.')
        list_of_word_count = [words_per_sentence(str(sentence)) for sentence in list_of_sentence]
        return(np.sum(list_of_word_count))

    def count_total_sentences(text):
        '''count total number of sentences in a text
           input: str or pandas.core.series.Series of text
           returns: integer'''
        if isinstance(text, pd.Series):
            text = ' '.join(text)
        sentences = sent_tokenize(text)
        return(len(sentences))

    # Count Syllables
    # https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
    def syllables(word):
        '''backup syllable counter if word not in NLTK-CMU dictionary'''
        #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        try:
            if word[0] in vowels:
                count +=1
        except:
            count += 0
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count

    d = cmudict.dict()
    def nsyl(word):
        '''input: string - word
           returns: integer count of syllables in word'''
        try:
            # needs the [0] otherwise words like 'of' returns [1,1]
            return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        except KeyError:
            #if word not found in cmudict
            return syllables(word)

    def count_total_syllables(text):
        '''count total number of sentences in a text
           input: str or pandas.core.series.Series of text
           returns: integer count'''
        if isinstance(text, str):
            list_of_sentence = [text]
        elif isinstance(text, pd.Series):
            list_of_sentence = text_to_sentence(text)
        else:
            print('count_total_syllables: Error: not a string or pandas series object.')
        list_of_words = text_to_wordlist(list_of_sentence)
        syllable_list = [nsyl(word) for word in list_of_words]
        return(np.sum(syllable_list))

    def count_of_letters(text):
        '''count total number of letters or digits in a text
           input: str or pandas.core.series.Series of text
           returns: integer count'''
        if isinstance(text, pd.Series):
            text = ' '.join(text)
        # Replace punctuations with an empty string.
        str1 = re.sub(r"[^\w\s]|_", "", text)
        no_spaces = str1.replace(" ", "")
        return(len(no_spaces))


    def difficult_words_list(list1):
        '''returns difference of list with easy_word list for Dale-Chall
           input: two lists of strings/words
           returns: list of unique words in both lists'''
        if isinstance(list1, pd.Series):
            list1 = ' '.join(text)
        if isinstance(list1, list):
            list1 = list1[0]
        try:
            easy_words_file = open('./word_lists/DaleChallEasyWordList.txt', 'r')
            easy_words = easy_words_file.read().split('\n')
        except E:
            print("Error reading easy words file", E)
        easy_words_file.close()
        easy_words = [word.lower() for word in easy_words]
        easy_words = set(easy_words)
        diff = [word.lower() for word in list1.split() if word.lower() not in easy_words]
        return(diff)

    def dc_difficult_word_count(text):
        '''Count of difficult words - those not in Dale-Chall Easy Word List
           input: str or pandas.core.series.Series of text
           returns: integer count of difficult words in text'''
        list_of_dc_difficult = difficult_words_list(text)
        return(len(list_of_dc_difficult))

    lemmatizer = WordNetLemmatizer()
    def gf_complex_word_count(text):
        '''Count of complex - >= 3 syllables with caveats
           input: str or pandas.core.series.Series of text
           returns: integer count of complex words in text'''
        if isinstance(text, pd.Series):
            text = ' '.join(text)
        if isinstance(text, list):
            text = str(text[0])
        text = [word.lower() for word in text.split()]
        lemma = [lemmatizer.lemmatize(word) for word in text]
        stem = [re.sub("(?:ing|ed|es|ly)$","",word) for word in text]
        syllable_list = [nsyl(word) for word in stem]
        complex_count = sum(x > 2 for x in syllable_list)
        return(complex_count)

    def smog_poly_count(text):
        '''counts number of words with 3 or more syllables
           input: str or pandas.core.series.Series of text
           returns: integer count of polysyllabic words in text'''
        if isinstance(text, pd.Series):
            text = ' '.join(text)
        if isinstance(text, list):
            text = str(text[0])
        text = [word.lower() for word in text.split()]
        syllable_list = [nsyl(word) for word in text]
        poly_count = sum(x > 2 for x in syllable_list)
        return(poly_count)


    ######################## READABILITY SCORES ########################

    # https://en.wikipedia.org/wiki/Automated_readability_index
    # 4.71(chars/word) + 0.5(words/sentence) - 21.43
    def ari(text):
        '''input: string of sentence
           returns: float ari score'''
        character_count = count_of_letters(text)
        word_count = count_total_words(text)
        sentence_count = count_total_sentences(text)
        ari = 4.71*(character_count/word_count) + 0.5*(word_count/sentence_count) - 21.43
        return(ari)

    # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    # Flesch–Kincaid grade level
    # 0.39(total words/total sentences) + 11.8(total syllables/total words) - 15.59
    def flesch_kincaid(text):
        '''input: string or pandas series of text, multiple sentences
           returns: float - flesh kincaid grade level score'''
        num_total_words = count_total_words(text)
        num_total_sentences = count_total_sentences(text)
        num_total_syllables = count_total_syllables(text)
        fkgl = 0.39*(num_total_words/num_total_sentences) + 11.8*(num_total_syllables/num_total_words) - 15.59
        return(fkgl)

    # https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
    # 0.0588(average number of letters per 100 words) - 0.296(average number of sentences per 100 words) - 15.8
    def coleman_liau(text):
        '''input: string or pandas series of text, multiple sentences
           returns: float - Coleman-Liau index'''
        character_count = count_of_letters(text)
        word_count = count_total_words(text)
        sentence_count = count_total_sentences(text)
        l = character_count/word_count*100
        s = sentence_count/word_count*100
        cl = 0.0588*l - 0.296*s - 15.8
        return(cl)

    # https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
    # 0.1579(100*difficult words/words) + 0.496(words/sentences)
    def dale_chall(text):
        '''input: string or pandas series of text, multiple sentences
           returns: float - Dale-Chall readability score'''
        difficult_words = dc_difficult_word_count(text)
        word_count = count_total_words(text)
        sentence_count = count_total_sentences(text)
        dc = 0.1579*(100*difficult_words/word_count) + 0.496*(word_count/sentence_count)
        return(dc)

    # https://en.wikipedia.org/wiki/Gunning_fog_index
    # 0.4[(words/sentence) + 100(complex words/words)]
    def gunning_fog(text):
        '''input: string or pandas series of text, multiple sentences
           returns: float - Gunning Fog index readability score'''
        complex_words = gf_complex_word_count(text)
        word_count = count_total_words(text)
        sentence_count = count_total_sentences(text)
        gf = 0.4*((word_count/sentence_count) + 100*(complex_words/word_count))
        return(gf)

    # https://en.wikipedia.org/wiki/SMOG
    # 1.043*sqrt(30*number polysylables/number sentences)+3.1291
    def smog(text):
        '''input: string or pandas series of text, multiple sentences
           returns: float - SMOG grade readability score'''
        poly_count = smog_poly_count(text)
        sentence_count = count_total_sentences(text)
        smog_score = 1.043*np.sqrt(30*poly_count/sentence_count) + 3.1291
        return(smog_score)
    
    text_df['char_count'] = text_df['text'].apply(count_of_letters)
    text_df['syl_count'] = text_df['text'].apply(count_total_syllables)
    text_df['word_count'] = text_df['text'].apply(count_total_words)
    text_df['char_per_word'] = text_df['char_count']/text_df['word_count']#
    text_df['syl_per_word'] = text_df['syl_count']/text_df['word_count']#
    text_df['sent_count'] = text_df['text'].apply(count_total_sentences)
    text_df['word_per_sent'] = text_df['word_count']/text_df['sent_count']#

    text_df['dc_word_count'] = text_df['text'].apply(dc_difficult_word_count)
    text_df['gf_word_count'] = text_df['text'].apply(gf_complex_word_count)
    text_df['poly_word_count'] = text_df['text'].apply(smog_poly_count)

    text_df['dc_word_perc'] = text_df['dc_word_count']/text_df['word_count']#
    text_df['gf_word_perc'] = text_df['gf_word_count']/text_df['word_count']#
    text_df['poly_word_perc'] = text_df['poly_word_count']/text_df['word_count']#

    text_df['ari'] = text_df['text'].apply(ari)
    text_df['flesch_kincaid'] = text_df['text'].apply(flesch_kincaid)
    text_df['coleman_liau'] = text_df['text'].apply(coleman_liau)
    text_df['dale_chall'] = text_df['text'].apply(dale_chall)
    text_df['gunning_fog'] = text_df['text'].apply(gunning_fog)
    text_df['smog'] = text_df['text'].apply(smog)
    
    print("Length of text_df", len(text_df))
    ########################### PARALLEL PHRASE COUNT ############################
    print('---------PARALLEL PHRASE COUNT---------')
    parser = CoreNLPParser()
    print_error = True
    def count_parallels(sents):
        count = 0
        for phrase in sents:
            try:
                sent, = parser.parse_text(phrase)
            except:
                #if print_error:
                #    print("   Be sure Stanford CoreNLP server started and parser instantiated!")
                #    print("   Errors also seem to occur with quotes")
                #    print_error = False
                continue
            poss = []
            words = []
            for word in sent.pos():
                poss.append(word[1])
                words.append(word[0])
            stop = False
            results = []
            for length in range(7,3,-1):
                length = min(length, len(words))
                for i in range(len(poss)-length+1):
                    for j in range(len(poss)-length+1):
                        if abs(i-j) > length:
                            if poss[i:i+length]==poss[j:j+length]:
                                if length > 4 or (',' not in poss[i:i+length] and '``' not in poss[i:i+length]):
                                    results.append([i,j,length])
                                    count += 1
                                    stop = True
                                    break
                    if stop: break
                if stop: break

        return(count)

    print('      ...pst, this is really slow')
    text_df['parallel_count'] = text_df.text.apply(count_parallels)
    text_df['parallel_per_sent'] = text_df.parallel_count/text_df.num_sents
    
    print("Length of text_df", len(text_df))
    ########################### PARSE TREE DEPTH ###########################
    print('---------PARSE TREE DEPTH----------')

    # Make data frame of sentences and parse tree depth of each
    def walk_tree_depth(node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(walk_tree_depth(child, depth+1) for child in node.children )
        else:
            return depth
        
    tree_depth = pd.DataFrame(columns = ['date', 'source', 'sentence', 'depth'])
    for i, speech in enumerate(text_df['sentences']):
        for j, sentence in enumerate(speech):
            doc = nlp_spacy(sentence)
            depth = [walk_tree_depth(sent.root, 0) for sent in doc.sents][0]
            tree_depth.loc[len(tree_depth)] = [text_df.index[i], text_df['source'].iloc[i], sentence, depth]

    mean_depth=tree_depth.groupby(by='date').mean(depth)
    text_df=pd.merge(text_df, mean_depth, how='left', on='date')
    tree_depth.to_csv('sentence_depth_'+suffix+'_sentence.csv',index=False)
    
    tsw = text_df[['text', 'sentences', 'words', 'word_set']]
    tsw.to_csv('text_sentence_words_'+suffix+'_sentence.csv')
    text_df.drop(['text', 'sentences', 'words', 'word_set'], axis=1, inplace=True)
    text_df.to_csv('tidy_data_'+suffix+'_sentence.csv')    


if __name__ == "__main__":
    main()