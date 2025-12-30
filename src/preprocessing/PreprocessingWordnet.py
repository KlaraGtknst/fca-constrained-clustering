import json
import nltk
from nltk.corpus import reuters, wordnet as wn

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
    word_to_synsets = {w: wn.synsets(w) for w in attribute_names}

    synsets = {}
    for ws_k, ws_v in word_to_synsets.items():
        ws_v1 = []
        if len(ws_v) != 0:
            for sn_set in ws_v:
                ws_v1.append(sn_set.lemma_names())
        synsets[ws_k] = ws_v1
    groups = terms_in_same_synset(synsets)
    anchestors = check_common_ancestors_all(synsets)
    data = {"synsets": synsets, "groups": groups, "anchestors": anchestors}
    with open("../../resources/wordnet_reuters.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('finished')

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

def check_common_ancestors_all(synsets):
    anchestors = {}
    for cat, syns in synsets.items():
        if len(syns) == 0:
            continue
        for cat2 in synsets.keys():
            if cat != cat2:
                candidate_result = check_common_ancestors(cat, cat2)
                if len(candidate_result) == 0:
                    continue
                anchestors[f'{cat}, {cat2}'] = candidate_result
    return anchestors   

def check_common_ancestors(word1: str, word2: str):
    """
    Check IS-A relation for a certain word-pair
    :param word1:
    :param word2:
    :return:
    """
    synsets1 = wn.synsets(word1, pos=wn.NOUN)  # Focus on nouns; adjust pos as needed
    synsets2 = wn.synsets(word2, pos=wn.NOUN)
    results = []
    for s1 in synsets1:
        for s2 in synsets2:
            dist = s1.shortest_path_distance(s2)
            if dist is not None:
                # Find common hypernyms (shared parents/ancestors)
                common = s1.common_hypernyms(s2)
                if common:
                    lca = s1.lowest_common_hypernyms(s2)[0]  # Lowest common ancestor
                    results.append({
                        'syn1': s1.name(), 'syn2': s2.name(),
                        'distance': dist, 'lca': lca.name(),
                        'common_hypernyms': [h.name() for h in common]
                    })
    return results


if __name__ == '__main__':
    main()