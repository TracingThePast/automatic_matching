import icu
import re
import numpy as np
from Levenshtein import distance as levenshtein_distance

latin_transliterator = icu.Transliterator.createInstance('Any-Latin; Latin-ASCII;IPA-XSampa;NFD; [:Nonspacing Mark:] Remove; NFC; Lower();')
# Creates a transliterator that replaces all non latin characters, removes all accents and lowercases the entire string.
# For testing see: https://icu4c-demos.unicode.org/icu-bin/translit

replacements = {
        'á': 'a',        'ï': 'i',        'ş': 's',        'ó': 'o',
        'ł': 'l',        'ñ': 'n',        'è': 'e',        'ç': 'c',
        'ß': 'ss',       'ô': 'o',        'ü': 'u',        'æ': 'ae',
        'ø': 'o',        'û': 'u',        'ã': 'a',        'ê': 'e',
        'ë': 'e',        'ù': 'u',        'ï': 'i',        'î': 'i',
        'é': 'e',        'í': 'i',        'ú': 'u',        'ý': 'y',
        'à': 'a',        'ì': 'i',        'ò': 'o',        'ã': 'a',
        'ñ': 'n',        'õ': 'o',        'ç': 'c',        'ă': 'a',
        'ā': 'a',        'ē': 'e',        'ī': 'i',        'ō': 'o',
        'ū': 'u',        'ȳ': 'y',        'ǎ': 'a',        'ě': 'e',
        'ǐ': 'i',        'ǒ': 'o',        'ǔ': 'u',        'ǜ': 'u',
        'ǽ': 'ae',       'ð': 'd',        'œ': 'oe',       'ẽ': 'e',
        'ỹ': 'y',        'ũ': 'u',        'ȩ': 'e',        'ȯ': 'o',
        'ḧ': 'h',        'ẅ': 'w',        'ẗ': 't',        'ḋ': 'd',
        'ẍ': 'x',        'ẁ': 'w',        'ẃ': 'w',        'ỳ': 'y',
        'ÿ': 'y',        'ỹ': 'y',        'ŷ': 'y',        'ą': 'a',
        'į': 'i',        'ś': 's',        'ź': 'z',        'ć': 'c',
        'ń': 'n',        'ę': 'e',        'ţ': 't',        'ģ': 'g',
        'ķ': 'k',        'ņ': 'n',        'ļ': 'l',        'ż': 'z',
        'ċ': 'c',        'š': 's',        'ž': 'z',        'ď': 'd',
        'ľ': 'l',        'ř': 'r',        'ǧ': 'g',        'ǳ': 'dz',
        'ǆ': 'dz',       'ǉ': 'lj',       'ǌ': 'nj',       'ǚ': 'u',
        'ǘ': 'u',        'ǜ': 'u',        'ǟ': 'a',        'ǡ': 'a',
        'ǣ': 'ae',       'ǥ': 'g',        'ǭ': 'o',        'ǯ': 'z',
        'ȟ': 'h',        'ȱ': 'o',        'ȹ': 'y',        'ḭ': 'i',
        'ḯ': 'i',        'ḱ': 'k'
    }

def test_transliteration():
    matches = 0
    for replacement in replacements:
        translit = latin_transliterator.transliterate(replacement)
        if translit != replacements[replacement]:
            print('Replacement mismatch', translit, replacement, replacements[replacement])
        else:
            matches += 1
    print(f'Found {matches} matches of ({len(replacements)})')

def normalize_string(value, is_surname = False):
    value = latin_transliterator.transliterate(value)
    if is_surname:
        value = re.sub(r'owa$|ova$', '', value)
        value = re.sub(r'sohns$', 'sons', value)
        value = re.sub(r'sohn$', 'son', value)
        value = re.sub(r'(?<=sk|ck)a$', 'i', value) # only for last names

    value = value.replace('ae', 'a').replace('oe', 'o')
    value = re.sub(r'(?<!a)ue', 'u', value)
    value = value.replace('th', 't').replace('ck', 'k').replace('ph', 'f')
    value = value.replace('j', 'i').replace('y', 'i').replace('w', 'v')
    value = re.sub(r'(?=h|q|s|z)c', 'k', value)
    if not is_surname:
        value = value.replace('tz', 'z')
    return re.sub(r'([a-zA-Z])\1', r'\1', value) # Remove double chararcters

def get_names_as_list(names, is_surname=False, remove_acronyms=True):
    names_split = names.replace(',', ' ').replace('-', ' ').replace('/', ' ').split(' ')
    result = []
    for name in names_split:
        name_length = len(name)
        if remove_acronyms:
            if  name_length == 1:
                continue
            if name_length == 2 and name[-1] == '.':
                continue
        if name_length > 0:
            result.append({
                "original": name,
                "normalized": normalize_string(name, is_surname)
            })
    return result

def match_against_local_data(local_data, external_data):
    '''
        Takes two lists of local and external values and compares them.
        Returns the mean value of all minimal Levenshtein distances and a ordered list all matched pairs.
    '''
    levenshtein_distances_normalized = []
    levenshtein_distances_original = []
    matched_pairs = []
    for external in external_data:
        temp_levenshtein_distances_normalized = []
        temp_levenshtein_distances_original = []
        for local in local_data:
            lvs_dist_normalized = levenshtein_distance(local['normalized'], external['normalized'])
            temp_levenshtein_distances_normalized.append(lvs_dist_normalized)
            lvs_dist_original = levenshtein_distance(local['original'], external['original'])
            temp_levenshtein_distances_original.append(lvs_dist_original)
            matched_pairs.append({
                'levenshtein_distance_normalized': lvs_dist_normalized,
                'levenshtein_distance_original': lvs_dist_original,
                'local': local,
                'external': external,
            })
            if lvs_dist_normalized == 0:
                # A perfect fit has been found, no further matching is necessary
                break
        levenshtein_distances_normalized.append(min(temp_levenshtein_distances_normalized))
        levenshtein_distances_original.append(min(temp_levenshtein_distances_original))
    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x['levenshtein_distance_normalized'])
    return {
        'mean_levenshtein_distance_normalized': np.mean(levenshtein_distances_normalized),
        'mean_levenshtein_distance_original': np.mean(levenshtein_distances_original),
        'levenshtein_distance_normalized': f'Mean: {np.mean(levenshtein_distances_normalized)}',
        'levenshtein_distance_original': f'Mean: {np.mean(levenshtein_distances_original)}',
        'matched_pairs': matched_pairs_sorted
    }

FORENAME_MAX_SCORE_CONTRIBUTION = 25
SURNAME_MAX_SCORE_CONTRIBUTION = 25

BIRTH_PLACE_MAX_SCORE_CONTRIBUTION = 15
BIRTH_DATE_MAX_SCORE_CONTRIBUTION = 15

DEATH_PLACE_MAX_SCORE_CONTRIBUTION = 10
DEATH_DATE_MAX_SCORE_CONTRIBUTION = 10

MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING = 66

TOTAL_MAX_SCORE_REACHABLE = FORENAME_MAX_SCORE_CONTRIBUTION + SURNAME_MAX_SCORE_CONTRIBUTION + BIRTH_PLACE_MAX_SCORE_CONTRIBUTION + BIRTH_DATE_MAX_SCORE_CONTRIBUTION + DEATH_PLACE_MAX_SCORE_CONTRIBUTION + DEATH_DATE_MAX_SCORE_CONTRIBUTION


def get_matching_score(local_data_set, external_data_set):
    '''
        Expected inputs: local_data_set and external_data_set:
        To get a complete match all values have to be provided.
        Expected layout e.g.:
        {
            'forenames': 'Anna',
            'surnames': 'Musterfrau', # (Include all surnames, birthnames etc. here, preferentially seperated by commas)
            'birth_place': 'München',
            'birth_date': 'DD.MM.YYYY',
            'death_place': 'Dachau',
            'death_date': 'DD.MM.YYYY',
        }
    '''
    results = {}
    absolute_score = 0
    absolute_score_original = 0
    max_score_reachable = 0

    # NAME information

    if 'forenames' in local_data_set and 'forenames' in external_data_set:
        local_forenames = get_names_as_list(local_data_set['forenames'])
        external_forenames = get_names_as_list(external_data_set['forenames'])
        if len(local_forenames) > 0 and len(external_forenames) > 0:
            forename_results = match_against_local_data(local_forenames, external_forenames)
            forename_score = FORENAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + forename_results['mean_levenshtein_distance_normalized']**4 * 0.05))
            forename_score_original = FORENAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + forename_results['mean_levenshtein_distance_original']**4 * 0.05))
            results['forename'] = forename_results
            results['forename']['score'] = forename_score
            results['forename']['score_original'] = forename_score_original
            max_score_reachable += FORENAME_MAX_SCORE_CONTRIBUTION
            absolute_score += forename_score
            absolute_score_original += forename_score_original

    if 'surnames' in local_data_set and 'surnames' in external_data_set:
        local_surnames = get_names_as_list(local_data_set['surnames'], is_surname=True)
        external_surnames = get_names_as_list(external_data_set['surnames'], is_surname=True)
        if len(local_surnames) > 0 and len(external_surnames) > 0:
            surname_results = match_against_local_data(local_surnames, external_surnames)
            surname_score = SURNAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + surname_results['mean_levenshtein_distance_normalized']**4 * 0.05))
            surname_score_original = SURNAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + surname_results['mean_levenshtein_distance_original']**4 * 0.05))
            results['surname'] = surname_results
            results['surname']['score'] = surname_score
            results['surname']['score_original'] = surname_score_original
            max_score_reachable += SURNAME_MAX_SCORE_CONTRIBUTION
            absolute_score += surname_score
            absolute_score_original += surname_score_original

    # BIRTH information
            
    if 'birth_place' in local_data_set and 'birth_place' in external_data_set:
        local_birth_place = get_names_as_list(local_data_set['birth_place'])
        external_birth_place = get_names_as_list(external_data_set['birth_place'])
        if len(local_birth_place) > 0 and len(external_birth_place) > 0:
            birth_place_results = match_against_local_data(local_birth_place, external_birth_place)
            if birth_place_results['matched_pairs'][0]['levenshtein_distance_normalized'] == 0:
                # For places if we get a perfect match on one of the entries we accept the entire matched string as a perfect match
                # Background often Place of Birth / Death might be formated differently e.g. localy: München / Bayern | externaly: München - Freistaat Bayern
                # even though this is obviously a match the score would be relatively small due tue the different formating
                birth_place_score = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
            else:
                birth_place_score = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + birth_place_results['mean_levenshtein_distance_normalized']**4 * 0.05))
            birth_place_score_original = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + birth_place_results['mean_levenshtein_distance_original']**4 * 0.05))
            results['birth_place'] = birth_place_results
            results['birth_place']['score'] = birth_place_score
            results['birth_place']['score_original'] = birth_place_score_original
            max_score_reachable += BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
            absolute_score += birth_place_score
            absolute_score_original += birth_place_score_original

    if 'birth_date' in local_data_set and 'birth_date' in external_data_set:
        if len(local_data_set['birth_date']) == 10 and len(external_data_set['birth_date']) == 10:
            birth_date_levenshtein_distance = levenshtein_distance(local_data_set['birth_date'], external_data_set['birth_date'])
            birth_date_score = BIRTH_DATE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + birth_date_levenshtein_distance**2) ) 
            results['birth_date'] = {
                'levenshtein_distance': birth_date_levenshtein_distance,
                'local': local_data_set['birth_date'],
                'external': external_data_set['birth_date'],
                'score': birth_date_score
            }
            max_score_reachable += BIRTH_DATE_MAX_SCORE_CONTRIBUTION
            absolute_score += birth_date_score
            absolute_score_original += birth_date_score

    # DEATH information

    if 'death_place' in local_data_set and 'death_place' in external_data_set:
        local_death_place = get_names_as_list(local_data_set['death_place'])
        external_death_place = get_names_as_list(external_data_set['death_place'])
        if len(local_death_place) > 0 and len(external_death_place) > 0:
            death_place_results = match_against_local_data(local_death_place, external_death_place)
            if death_place_results['matched_pairs'][0]['levenshtein_distance_normalized'] == 0:
                death_place_score = DEATH_PLACE_MAX_SCORE_CONTRIBUTION
            else:
                death_place_score = DEATH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + death_place_results['mean_levenshtein_distance']**4 * 0.05))
            death_place_score_original = DEATH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + death_place_results['mean_levenshtein_distance_original']**4 * 0.05))
            results['death_place'] = death_place_results
            results['death_place']['score'] = death_place_score
            results['death_place']['score_original'] = death_place_score_original
            max_score_reachable += DEATH_PLACE_MAX_SCORE_CONTRIBUTION
            absolute_score += death_place_score
            absolute_score_original += death_place_score_original
    

    if 'death_date' in local_data_set and 'death_date' in external_data_set:
        if len(local_data_set['death_date']) == 10 and len(external_data_set['death_date']) == 10:
            death_date_levenshtein_distance = levenshtein_distance(local_data_set['death_date'], external_data_set['death_date'])
            death_date_score = DEATH_DATE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + death_date_levenshtein_distance**2) ) 
            results['death_date'] = {
                'levenshtein_distance': death_date_levenshtein_distance,
                'local': local_data_set['death_date'],
                'external': external_data_set['death_date'],
                'score': death_date_score
            }
            max_score_reachable += DEATH_DATE_MAX_SCORE_CONTRIBUTION
            absolute_score += death_date_score
            absolute_score_original += death_date_score

    relative_score = ( absolute_score / max_score_reachable ) if max_score_reachable > 0 else 0
    total_score = (max_score_reachable / TOTAL_MAX_SCORE_REACHABLE ) * relative_score * 100
    relative_score_original = ( absolute_score_original / max_score_reachable ) if max_score_reachable > 0 else 0
    total_score_original = (max_score_reachable / TOTAL_MAX_SCORE_REACHABLE ) * relative_score_original * 100

    return {
        **results,
        'absolute_score': absolute_score,
        'relative_score': relative_score,
        'total_score': total_score,
        'absolute_score_original': absolute_score_original,
        'relative_score_original': relative_score_original,
        'total_score_original': total_score_original,
        'automatically_matched': total_score > MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING
    }


def get_results_as_html(result):
    style = "background-color: green;" if result["automatically_matched"] else "background-color: red;"
    automatched_message = " Automatically matched" if result["automatically_matched"] else ''
    output = f'<h1>Total Score: <b style="padding:4px; color: white; border-radius:4px;{style}">{result["total_score"]}</b>{automatched_message}</h1>'
    output += f'<p>Minimum score for automatching: {MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING}</p>'
    table_header = '<table><tr><th>Local value</th><th>External value</th><th>Levenshtein distance</th></tr>'
    if 'forename' in result:
        output += f'<h2>Forename:</h2><p>Score: <b>{result["forename"]["score"]} of {FORENAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["forename"]["mean_levenshtein_distance_normalized"]} ({result["forename"]["mean_levenshtein_distance_original"]})<b></p>{table_header}'
        for value_pair in result['forename']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance_normalized"]} ({value_pair["levenshtein_distance_original"]})</td></tr>'
        output += '</table>'
        
    if 'surname' in result:
        output += f'<h2>Surname:</h2><p>Score: <b>{result["surname"]["score"]} of {SURNAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["surname"]["mean_levenshtein_distance_normalized"]} ({result["surname"]["mean_levenshtein_distance_original"]})<b></p>{table_header}'
        for value_pair in result['surname']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance_normalized"]} ({value_pair["levenshtein_distance_original"]})</td></tr>'
        output += '</table>'
    
    if 'birth_date' in result:
        output += f'<h2>Date of birth:</h2><p>Score: <b>{result["birth_date"]["score"]} of {BIRTH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["birth_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["birth_date"]["local"]}</td><td>{result["birth_date"]["external"]}</td><td>{result["birth_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'birth_place' in result:
        output += f'<h2>Place of birth:</h2><p>Score: <b>{result["birth_place"]["score"]} of {BIRTH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["birth_place"]["mean_levenshtein_distance_normalized"]} ({result["birth_place"]["mean_levenshtein_distance_original"]})<b></p>{table_header}'
        for value_pair in result['birth_place']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance_normalized"]} ({value_pair["levenshtein_distance_original"]})</td></tr>'
        output += '</table>'

    if 'death_date' in result:
        output += f'<h2>Date of death:</h2><p>Score: <b>{result["death_date"]["score"]} of {DEATH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["death_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["death_date"]["local"]}</td><td>{result["death_date"]["external"]}</td><td>{result["death_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'death_place' in result:
        output += f'<h2>Place of death:</h2><p>Score: <b>{result["death_place"]["score"]} of {DEATH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["death_place"]["mean_levenshtein_distance_normalized"]} ({result["death_place"]["mean_levenshtein_distance_original"]})<b></p>{table_header}'
        for value_pair in result['death_place']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance_normalized"]} ({value_pair["levenshtein_distance_original"]})</td></tr>'
        output += '</table>'

    return output


