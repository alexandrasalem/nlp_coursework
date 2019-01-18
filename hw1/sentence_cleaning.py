from nltk import tokenize
from nltk import corpus
from collections import defaultdict
import string

punct_set = set(string.punctuation)
stop_words = [t.upper() for t in corpus.stopwords.words('english')]


with open("deserialized.txt", "r") as input_file:
	with open("deserialized_cleaned.txt", "w") as output_file:
		for line in input_file:
			line = line.replace('\n',' ')
			output_file.write(line)



def tokenizer(output_file_name, bool_remove_stopwords):
	print(output_file_name + ": ")
	with open("deserialized_cleaned.txt", "r") as input_file:
		with open(output_file_name, "w") as output_file:
			for line in input_file:
				sentences = tokenize.sent_tokenize(str(line))
				word_tokenized_sentences = []
				for sentence in sentences:
					word_tokenized_sentence = tokenize.word_tokenize(sentence)
					word_tokenized_sentence = [t for t in word_tokenized_sentence if t not in punct_set]
					word_tokenized_sentence = [t.upper() for t in word_tokenized_sentence]
					if bool_remove_stopwords:
						word_tokenized_sentence = [t for t in word_tokenized_sentence if t not in stop_words]
					output_file.write(" ".join(word_tokenized_sentence) + "\n")
					word_tokenized_sentences.append(word_tokenized_sentence)
				print("number of sentences: " + str(len(word_tokenized_sentences)))

tokenizer("tokenized_with_stopwords.txt", False)
tokenizer("tokenized_without_stopwords.txt", True)