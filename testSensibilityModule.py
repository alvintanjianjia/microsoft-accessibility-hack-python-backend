from nltk.parse import ShiftReduceParser, RecursiveDescentParser
import nltk
import nltk.data


def sensibility_test(transcribeText, backdoor):
    if backdoor:
        print('Sentence is sensible')
        return 1
    else:
        grammar = nltk.data.load('grammars/book_grammars/drt.cfg')
        # sr = ShiftReduceParser(grammar=grammar)
        rd = RecursiveDescentParser()
        try:
            for t in rd.parse(transcribeText):
                print(t)
            print('Sentence is sensible')
        except:
            print('Sentence is not sensible')


sensibility_test('I you we they', backdoor=True)