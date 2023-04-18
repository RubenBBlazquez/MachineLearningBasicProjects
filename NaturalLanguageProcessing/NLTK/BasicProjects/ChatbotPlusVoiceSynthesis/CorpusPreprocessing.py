import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import pyttsx3


def greetings(sentence):
    greetings_inputs = ('hola', 'buenas', 'saludos', 'qué tal', 'hey', 'buenos dias', 'tengo una duda')
    greetings_responses = ['Cómo puedo ayudarte?', 'hola encantado de atenderte, que te ocurre?', 'que te ocurre?']

    for word in sentence.split():
        if word.lower() in greetings_inputs:
            return random.choice(greetings_responses)


class CorpusPreprocessing:
    def __init__(self, file_path):
        self.corpus_text = open(file_path).read()
        self.corpus_sentences = []
        self.corpus_tokens = []
        self.lemmer = nltk.stem.WordNetLemmatizer()
        self.punctuations_dict = dict((ord(punctuation), None) for punctuation in string.punctuation)

        self.init_preprocessing()

    def init_preprocessing(self):
        self.corpus_sentences = nltk.sent_tokenize(self.corpus_text)
        self.corpus_tokens = nltk.word_tokenize(self.corpus_text)

    def lem_tokens(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def lem_normalize(self, text):
        return self.lem_tokens(nltk.word_tokenize(text.lower().translate(self.punctuations_dict)))

    def preprocessing_user_response(self, user_response: str):
        self.corpus_sentences.append(user_response)  # we add to corpus the final user response

        t_fiz_vect = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words=stopwords.words('spanish'))
        t_fiz_vect = t_fiz_vect.fit_transform(self.corpus_sentences)

        # now we evaluate the similarity between user message (t_fiz_vect[:-1]) and the corpus(t_fiz_vec)
        vals = cosine_similarity(t_fiz_vect[-1], t_fiz_vect)
        idx = vals.argsort()[0][-2]

        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if req_tfidf == 0:
            return 'I`m sorry, I dont understand you. If I can`t answer your question contact with the page team'

        return self.corpus_sentences[idx]

    def get_chatbot_message(self, user_response, include_robot_name=True):
        message = 'Ruben: ' if include_robot_name else ''
        user_response = user_response.lower()

        greeting = greetings(user_response)

        if greeting is not None:
            return message + greeting

        message += self.preprocessing_user_response(user_response)
        self.corpus_sentences.remove(user_response)

        return message


class SynthesizerMethods:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)
        self.engine.setProperty('voice', 'spanish')

    def talk(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == '__main__':
    corpus = CorpusPreprocessing('files/restaurant_corpus.txt')
    first_message = 'Hola'
    second_message = 'hay vinos?'

    print(first_message)
    print(corpus.get_chatbot_message(first_message))
    print(second_message)
    print(corpus.get_chatbot_message(second_message))

    synthesizer = SynthesizerMethods()
    synthesizer.talk(corpus.get_chatbot_message(first_message, include_robot_name=False))
    synthesizer.talk(corpus.get_chatbot_message(second_message, include_robot_name=False))
