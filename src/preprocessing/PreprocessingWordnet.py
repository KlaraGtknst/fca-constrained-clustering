import json
import nltk
from nltk.corpus import reuters, wordnet

'''
Wordnet Preprocessing: 
1) get category terms from reuters dataset
2) extract available synsets for category terms - relevant category synset
3) collect category terms if they occur within the same synset - (remove synsets duplicates)
4) extract related IS-A relations for relevant category synsets, check for circular dependencies (log as circular 
errors)
5) collect IS-A relations between relevant category synsets

Author: S.Schneider
'''


nltk.download('wordnet')
nltk.download('reuters')

def main():
    attribute_names = [str(c).lower() for c in reuters.categories()]
    word_to_synsets = {w: wordnet.synsets(w) for w in attribute_names}

    synsets = {}
    for ws_k, ws_v in word_to_synsets.items():
        ws_v1 = []
        if len(ws_v) != 0:
            for sn_set in ws_v:
                ws_v1.append(sn_set.lemma_names())
        synsets[ws_k] = ws_v1
    groups = terms_in_same_synset(synsets)
    data = {"synsets": synsets, "groups": groups}
    with open("../../resources/wordnet_reuters.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    pass

def terms_in_same_synset(synsets):
    synset_to_words = {}
    for k_cat in synsets.keys():
        for k_cat2, v_synsets in synsets.items():
            if k_cat == k_cat2:
                continue
            for vs in v_synsets:
                if k_cat in vs:
                    synset_to_words[k_cat] = (k_cat2, vs)
    return synset_to_words
if __name__ == '__main__':
    main()