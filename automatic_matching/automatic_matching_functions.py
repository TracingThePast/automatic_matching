import icu
import re
import numpy as np
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio

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


def get_names_as_list_flattend(values, is_surname = False):
    result = []
    list_values = get_names_as_list(values, is_surname)
    for val in list_values:
        result.append(val['original'])
        result.append(val['normalized'])
    return ', '.join(result)

def match_against_local_data(local_data, external_data):
    '''
        Takes two lists of local and external values and compares them.
        Returns the mean value of all minimal Levenshtein ratio and a ordered list all matched pairs.
    '''
    matched_pairs = []
    larger_data_set = external_data
    larger_data_set_label = 'external'
    smaller_data_set = local_data
    smaller_data_set_label = 'local'
    if len(local_data) > len(external_data):
        larger_data_set = local_data
        larger_data_set_label = 'local'
        smaller_data_set = external_data
        smaller_data_set_label = 'external'

    names_in_smaller_set_normalized = {}
    names_in_smaller_set_original = {}
    for smaller in smaller_data_set:
        names_in_smaller_set_normalized[smaller['normalized']] = []
        names_in_smaller_set_original[smaller['original']] = []

    for smaller in smaller_data_set:
        temp_levenshtein_ratio_normalized = []
        temp_levenshtein_ratio_original = []
        for larger in larger_data_set:
            lvs_ratio_normalized = levenshtein_ratio(smaller['normalized'], larger['normalized'])
            temp_levenshtein_ratio_normalized.append(lvs_ratio_normalized)
            lvs_ratio_original = levenshtein_ratio(smaller['original'], larger['original'])
            temp_levenshtein_ratio_original.append(lvs_ratio_original)
            matched_pairs.append({
                'levenshtein_ratio_normalized': lvs_ratio_normalized,
                'levenshtein_ratio_original': lvs_ratio_original,
                larger_data_set_label: larger,
                smaller_data_set_label: smaller
            })
            if lvs_ratio_normalized == 1:
                # A perfect fit has been found, no further matching is necessary
                break
        # As some of the names might occur more than once especially after normalization
        names_in_smaller_set_normalized[smaller['normalized']].append(max(temp_levenshtein_ratio_normalized))
        names_in_smaller_set_original[smaller['original']].append(max(temp_levenshtein_ratio_original))
        
    levenshtein_ratios_normalized = []
    levenshtein_ratios_orginal = []

    for smaller_name_normalized in names_in_smaller_set_normalized:
        tmp_val = 0
        if len(names_in_smaller_set_normalized[smaller_name_normalized]) > 0:
            tmp_val = np.mean(names_in_smaller_set_normalized[smaller_name_normalized])**2
            if max(names_in_smaller_set_normalized[smaller_name_normalized]) == 1:
                tmp_val = 1
        levenshtein_ratios_normalized.append(tmp_val)
    for smaller_name_original in names_in_smaller_set_original:
        tmp_val = 0
        if len(names_in_smaller_set_original[smaller_name_original]) > 0:
            tmp_val = np.mean(names_in_smaller_set_original[smaller_name_original])**2
            if max(names_in_smaller_set_normalized[smaller_name_normalized]) == 1:
                tmp_val = 1
        levenshtein_ratios_orginal.append(tmp_val)
        


    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: -x['levenshtein_ratio_normalized'])
    mean_levenshtein_ratio_normalized = np.mean(levenshtein_ratios_normalized)
    median_levenshtein_ratio_normalized = np.median(levenshtein_ratios_normalized)
    mean_levenshtein_ratio_original = np.mean(levenshtein_ratios_orginal)
    return {
        'mean_levenshtein_ratio_normalized': mean_levenshtein_ratio_normalized,
        'median_levenshtein_ratio_normalized': median_levenshtein_ratio_normalized,
        'mean_levenshtein_ratio_original': mean_levenshtein_ratio_original,
        'levenshtein_ratio_normalized': f'Mean: {mean_levenshtein_ratio_normalized}',
        'levenshtein_ratio_original': f'Mean: {mean_levenshtein_ratio_original}',
        'matched_pairs': matched_pairs_sorted,
        'score': (mean_levenshtein_ratio_normalized + median_levenshtein_ratio_normalized) / 2
    }

def match_date_against_local_date(local_date, external_date):
    match_score = 0
    date_levenshtein_distance = levenshtein_distance(local_date, external_date)
    best_match_found = local_date
    day_month_switched = False
    if date_levenshtein_distance > 0:
        try: 
            local_day, local_month, local_year = local_date.split('.')
            external_day, external_month, external_year = external_date.split('.')
            # first check for switched month and day
            day_month_switched_levenshtein_distance = -1
            day_month_levenshtein_distance = levenshtein_distance(f'{local_day}.{local_month}', f'{external_day}.{external_month}')
            if day_month_levenshtein_distance > 0:
                if max([int(external_month), int(external_day), int(local_month), int(local_day)]) < 13:
                    day_month_switched_levenshtein_distance = levenshtein_distance(f'{local_day}.{local_month}', f'{external_month}.{external_day}')
                    if day_month_switched_levenshtein_distance == 0:
                        day_month_switched = True

            year_levenshtein_distance = levenshtein_distance(local_year, external_year)
            if year_levenshtein_distance == 0:
                # The years match
                if day_month_switched_levenshtein_distance == 0:
                    match_score = ( 1 + 0.85 ) / 2
                else:
                    match_score = ( 1 + 0.85 / ( 1 + day_month_levenshtein_distance**2) ) / 2
            else:
                # Two numbers in date have been edited
                # The last two digits in the year might have been switched
                year_switched_last_levenshtein_distance = levenshtein_distance(local_year, f'{external_year[0]}{external_year[1]}{external_year[3]}{external_year[2]}')
                year_contribution = 0.75
                year_distance = abs(int(local_year) - int(external_year))
                if year_switched_last_levenshtein_distance == 0:
                    year_contribution = 0.90
                else:
                    year_contribution = 0.75 - year_distance / 10
                if year_contribution <= 0:
                    match_score = 0
                else:
                    if day_month_levenshtein_distance == 0:
                        # Only the years have altercations or misspellings
                        match_score = ( year_contribution + 1 ) / 2
                    elif day_month_switched_levenshtein_distance == 1:
                        # The years have altercations or misspellings and date and month have been switched
                        match_score = ( year_contribution + 0.85 ) / 2
                    else:
                        match_score = ( year_contribution + 0.85 / ( 1 + day_month_levenshtein_distance**2) ) / 2

        except:
            match_score = 0
    else:
        match_score = 1

    return {
        'levenshtein_distance': date_levenshtein_distance,
        'score': match_score,
        'day_month_switched': day_month_switched
    }


FORENAME_MAX_SCORE_CONTRIBUTION = 25
FORENAME_MIN_SCORE = 7.5
SURNAME_MAX_SCORE_CONTRIBUTION = 25
SURNAME_MIN_SCORE = 7.5

BIRTH_PLACE_MAX_SCORE_CONTRIBUTION = 10
BIRTH_DATE_MAX_SCORE_CONTRIBUTION = 20

DEATH_PLACE_MAX_SCORE_CONTRIBUTION = 10
DEATH_DATE_MAX_SCORE_CONTRIBUTION = 10

MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING = 60

TOTAL_MAX_SCORE_REACHABLE = FORENAME_MAX_SCORE_CONTRIBUTION + SURNAME_MAX_SCORE_CONTRIBUTION + BIRTH_PLACE_MAX_SCORE_CONTRIBUTION + BIRTH_DATE_MAX_SCORE_CONTRIBUTION + DEATH_PLACE_MAX_SCORE_CONTRIBUTION + DEATH_DATE_MAX_SCORE_CONTRIBUTION


def get_matching_score(local_data_set, external_data_set):
    '''
        Expected inputs: local_data_set and external_data_set:
        To get a complete match all values have to be provided.
        Expected layout e.g.:
        {
            'forenames': 'Anna',
            'surnames': 'Musterfrau', # (Include all surnames and birthnames, etc. here, preferentially seperated by commas)
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
    name_min_score_reached = True

    # NAME information

    if 'forenames' in local_data_set and 'forenames' in external_data_set:
        local_forenames = get_names_as_list(local_data_set['forenames'])
        external_forenames = get_names_as_list(external_data_set['forenames'])
        if len(local_forenames) > 0 and len(external_forenames) > 0:
            forename_results = match_against_local_data(local_forenames, external_forenames)
            forename_score = FORENAME_MAX_SCORE_CONTRIBUTION * forename_results['score']**2
            forename_score_original = FORENAME_MAX_SCORE_CONTRIBUTION * forename_results['mean_levenshtein_ratio_original']**2
            results['forename'] = forename_results
            results['forename']['score'] = forename_score
            results['forename']['score_original'] = forename_score_original
            max_score_reachable += FORENAME_MAX_SCORE_CONTRIBUTION
            absolute_score += forename_score
            absolute_score_original += forename_score_original
            if forename_score < FORENAME_MIN_SCORE:
                name_min_score_reached = False

    if 'surnames' in local_data_set and 'surnames' in external_data_set:
        local_surnames = get_names_as_list(local_data_set['surnames'], is_surname=True)
        external_surnames = get_names_as_list(external_data_set['surnames'], is_surname=True)
        if len(local_surnames) > 0 and len(external_surnames) > 0:
            surname_results = match_against_local_data(local_surnames, external_surnames)
            surname_score = SURNAME_MAX_SCORE_CONTRIBUTION * surname_results['score']**2
            surname_score_original = SURNAME_MAX_SCORE_CONTRIBUTION * surname_results['mean_levenshtein_ratio_original']**2
            results['surname'] = surname_results
            results['surname']['score'] = surname_score
            results['surname']['score_original'] = surname_score_original
            max_score_reachable += SURNAME_MAX_SCORE_CONTRIBUTION
            absolute_score += surname_score
            absolute_score_original += surname_score_original
            if surname_score < SURNAME_MIN_SCORE:
                name_min_score_reached = False

    # BIRTH information
            
    if 'birth_place' in local_data_set and 'birth_place' in external_data_set:
        local_birth_place = get_names_as_list(local_data_set['birth_place'])
        external_birth_place = get_names_as_list(external_data_set['birth_place'])
        if len(local_birth_place) > 0 and len(external_birth_place) > 0:
            birth_place_results = match_against_local_data(local_birth_place, external_birth_place)
            if birth_place_results['matched_pairs'][0]['levenshtein_ratio_normalized'] == 1:
                # For places if we get a perfect match on one of the entries we accept the entire matched string as a perfect match
                # Background often Place of Birth / Death might be formated differently e.g. localy: München / Bayern | externaly: München - Freistaat Bayern
                # even though this is obviously a match the score would be relatively small due tue the different formating
                birth_place_score = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
            else:
                birth_place_score = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION * birth_place_results['mean_levenshtein_ratio_normalized']
            birth_place_score_original = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION * birth_place_results['mean_levenshtein_ratio_original']
            results['birth_place'] = birth_place_results
            results['birth_place']['score'] = birth_place_score
            results['birth_place']['score_original'] = birth_place_score_original
            max_score_reachable += BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
            absolute_score += birth_place_score
            absolute_score_original += birth_place_score_original

    if 'birth_date' in local_data_set and 'birth_date' in external_data_set:
        if len(local_data_set['birth_date']) == 10 and len(external_data_set['birth_date']) == 10:
            birth_date_result = match_date_against_local_date(local_data_set['birth_date'], external_data_set['birth_date'])
            birth_date_score = BIRTH_DATE_MAX_SCORE_CONTRIBUTION * birth_date_result['score']
            # Added a penalty for massive deviations in the year

            results['birth_date'] = {
                'levenshtein_distance': birth_date_result['levenshtein_distance'],
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
            if death_place_results['matched_pairs'][0]['levenshtein_ratio_normalized'] == 1:
                death_place_score = DEATH_PLACE_MAX_SCORE_CONTRIBUTION
            else:
                death_place_score = DEATH_PLACE_MAX_SCORE_CONTRIBUTION * death_place_results['mean_levenshtein_ratio_normalized']
            death_place_score_original = DEATH_PLACE_MAX_SCORE_CONTRIBUTION * death_place_results['mean_levenshtein_ratio_original']
            results['death_place'] = death_place_results
            results['death_place']['score'] = death_place_score
            results['death_place']['score_original'] = death_place_score_original
            max_score_reachable += DEATH_PLACE_MAX_SCORE_CONTRIBUTION
            absolute_score += death_place_score
            absolute_score_original += death_place_score_original
    

    if 'death_date' in local_data_set and 'death_date' in external_data_set:
        if len(local_data_set['death_date']) == 10 and len(external_data_set['death_date']) == 10:
            death_date_result = match_date_against_local_date(local_data_set['death_date'], external_data_set['death_date'])
            death_date_score = DEATH_DATE_MAX_SCORE_CONTRIBUTION * death_date_result['score']
            # Added a penalty for massive deviations in the year

            results['death_date'] = {
                'levenshtein_distance': death_date_result['levenshtein_distance'],
                'local': local_data_set['death_date'],
                'external': external_data_set['death_date'],
                'score': death_date_score
            }
            max_score_reachable += DEATH_DATE_MAX_SCORE_CONTRIBUTION
            absolute_score += death_date_score
            absolute_score_original += death_date_score

    if not name_min_score_reached:
        absolute_score -= SURNAME_MAX_SCORE_CONTRIBUTION
    relative_score = ( absolute_score / max_score_reachable ) if max_score_reachable > 0 else 0
    total_score = (max_score_reachable / TOTAL_MAX_SCORE_REACHABLE ) * relative_score * 100
    relative_score_original = ( absolute_score_original / max_score_reachable ) if max_score_reachable > 0 else 0
    total_score_original = (max_score_reachable / TOTAL_MAX_SCORE_REACHABLE ) * relative_score_original * 100

    return {
        **results,
        'absolute_score': absolute_score,
        'relative_score': relative_score,
        'total_score': total_score,
        'name_min_score_reached': name_min_score_reached,
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
    table_header = '<table><tr><th>Local value</th><th>External value</th><th>Levenshtein ratio</th></tr>'
    if 'forename' in result:
        output += f'<h2>Forename:</h2><p>Score: <b>{result["forename"]["score"]} of {FORENAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["forename"]["mean_levenshtein_ratio_normalized"]} ({result["forename"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
        for value_pair in result['forename']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
        output += '</table>'
        
    if 'surname' in result:
        output += f'<h2>Surname:</h2><p>Score: <b>{result["surname"]["score"]} of {SURNAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["surname"]["mean_levenshtein_ratio_normalized"]} ({result["surname"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
        for value_pair in result['surname']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
        output += '</table>'
    
    if 'birth_date' in result:
        output += f'<h2>Date of birth:</h2><p>Score: <b>{result["birth_date"]["score"]} of {BIRTH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["birth_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["birth_date"]["local"]}</td><td>{result["birth_date"]["external"]}</td><td>{result["birth_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'birth_place' in result:
        output += f'<h2>Place of birth:</h2><p>Score: <b>{result["birth_place"]["score"]} of {BIRTH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["birth_place"]["mean_levenshtein_ratio_normalized"]} ({result["birth_place"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
        for value_pair in result['birth_place']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
        output += '</table>'

    if 'death_date' in result:
        output += f'<h2>Date of death:</h2><p>Score: <b>{result["death_date"]["score"]} of {DEATH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["death_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["death_date"]["local"]}</td><td>{result["death_date"]["external"]}</td><td>{result["death_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'death_place' in result:
        output += f'<h2>Place of death:</h2><p>Score: <b>{result["death_place"]["score"]} of {DEATH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["death_place"]["mean_levenshtein_ratio_normalized"]} ({result["death_place"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
        for value_pair in result['death_place']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
        output += '</table>'

    return output


