import numpy as np
from Levenshtein import distance as levenshtein_distance

def get_names_as_list(names, remove_acronyms=True):
    names_split = names.replace(',', ' ').split(' ')
    result = []
    for name in names_split:
        name_length = len(name)
        if remove_acronyms:
            if  name_length == 1:
                continue
            if name_length == 2 and name[-1] == '.':
                continue
        if name_length > 0:
            result.append(name)
    return result

def match_against_local_data(local_data, external_data):
    '''
        Takes two lists of local and external values and compares them.
        Returns the mean value of all minimal Levenshtein distances and a ordered list all matched pairs.
    '''
    levenshtein_distances = []
    matched_pairs = []
    for external in external_data:
        temp_levenshtein_distances = []
        for local in local_data:
            lvs_dist = levenshtein_distance(local, external)
            temp_levenshtein_distances.append(lvs_dist)
            matched_pairs.append({
                'levenshtein_distance': lvs_dist,
                'local': local,
                'external': external
            })
            if lvs_dist == 0:
                # A perfect fit has been found, no further matching is necessary
                break
        levenshtein_distances.append(min(temp_levenshtein_distances))
    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: x['levenshtein_distance'])
    return {
        'mean_levenshtein_distance': np.mean(levenshtein_distances),
        'matched_pairs': matched_pairs_sorted
    }

FORENAME_MAX_SCORE_CONTRIBUTION = 25
SURNAME_MAX_SCORE_CONTRIBUTION = 25

BIRTH_PLACE_MAX_SCORE_CONTRIBUTION = 15
BIRTH_DATE_MAX_SCORE_CONTRIBUTION = 15

DEATH_PLACE_MAX_SCORE_CONTRIBUTION = 10
DEATH_DATE_MAX_SCORE_CONTRIBUTION = 10

MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING = 75

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
    total_score = 0

    # NAME information

    if 'forenames' in local_data_set and 'forenames' in external_data_set:
        local_forenames = get_names_as_list(local_data_set['forenames'])
        external_forenames = get_names_as_list(external_data_set['forenames'])
        if len(local_forenames) > 0 and len(external_forenames) > 0:
            forename_results = match_against_local_data(local_forenames, external_forenames)
            forename_score = FORENAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + forename_results['mean_levenshtein_distance']**4 * 0.05))
            results['forename'] = forename_results
            results['forename']['score'] = forename_score
            total_score += forename_score

    if 'surnames' in local_data_set and 'surnames' in external_data_set:
        local_surnames = get_names_as_list(local_data_set['surnames'])
        external_surnames = get_names_as_list(external_data_set['surnames'])
        if len(local_surnames) > 0 and len(external_surnames) > 0:
            surname_results = match_against_local_data(local_surnames, external_surnames)
            surname_score = SURNAME_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + surname_results['mean_levenshtein_distance']**4 * 0.05))
            results['surname'] = surname_results
            results['surname']['score'] = surname_score
            total_score += surname_score

    # BIRTH information
            
    if 'birth_place' in local_data_set and 'birth_place' in external_data_set:
        local_birth_place = get_names_as_list(local_data_set['birth_place'])
        external_birth_place = get_names_as_list(external_data_set['birth_place'])
        if len(local_birth_place) > 0 and len(external_birth_place) > 0:
            birth_place_results = match_against_local_data(local_birth_place, external_birth_place)
            birth_place_score = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + birth_place_results['mean_levenshtein_distance']**4 * 0.05))
            results['birth_place'] = birth_place_results
            results['birth_place']['score'] = birth_place_score
            total_score += birth_place_score

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
            total_score += birth_date_score

    # DEATH information

    if 'death_place' in local_data_set and 'death_place' in external_data_set:
        local_death_place = get_names_as_list(local_data_set['death_place'])
        external_death_place = get_names_as_list(external_data_set['death_place'])
        if len(local_death_place) > 0 and len(external_death_place) > 0:
            death_place_results = match_against_local_data(local_death_place, external_death_place)
            death_place_score = DEATH_PLACE_MAX_SCORE_CONTRIBUTION * ( 1 / ( 1 + death_place_results['mean_levenshtein_distance']**4 * 0.05))
            results['death_place'] = death_place_results
            results['death_place']['score'] = death_place_score
            total_score += death_place_score
    

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
            total_score += death_date_score

    return {
        **results,
        'total_score': total_score,
        'automatically_matched': total_score > MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING
    }


def get_results_as_html(result):
    style = "background-color: green;" if result["automatically_matched"] else "background-color: red;"
    automatched_message = " Automatically matched" if result["automatically_matched"] else ''
    output = f'<h1>Total Score: <b style="padding:4px; color: white; border-radius:4px;{style}">{result["total_score"]}</b>{automatched_message}</h1>'
    output += f'<p>Minimum score for automatching: {MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING}</p>'
    table_header = '<table><tr><th>Local value</th><th>External value</th><th>Levenshtein distance</th></tr>'
    if 'forename' in result:
        output += f'<h2>Forename:</h2><p>Score: <b>{result["forename"]["score"]} of {FORENAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["forename"]["mean_levenshtein_distance"]}<b></p>{table_header}'
        for value_pair in result['forename']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance"]}</td></tr>'
        output += '</table>'
        
    if 'surname' in result:
        output += f'<h2>Surname:</h2><p>Score: <b>{result["surname"]["score"]} of {SURNAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["surname"]["mean_levenshtein_distance"]}<b></p>{table_header}'
        for value_pair in result['surname']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance"]}</td></tr>'
        output += '</table>'
    
    if 'birth_date' in result:
        output += f'<h2>Date of birth:</h2><p>Score: <b>{result["birth_date"]["score"]} of {BIRTH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["birth_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["birth_date"]["local"]}</td><td>{result["birth_date"]["external"]}</td><td>{result["birth_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'birth_place' in result:
        output += f'<h2>Place of birth:</h2><p>Score: <b>{result["birth_place"]["score"]} of {BIRTH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["birth_place"]["mean_levenshtein_distance"]}<b></p>{table_header}'
        for value_pair in result['forename']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'death_date' in result:
        output += f'<h2>Date of death:</h2><p>Score: <b>{result["death_date"]["score"]} of {DEATH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["death_date"]["levenshtein_distance"]}<b></p>{table_header}'
        output += f'<tr><td>{result["death_date"]["local"]}</td><td>{result["death_date"]["external"]}</td><td>{result["death_date"]["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    if 'death_place' in result:
        output += f'<h2>Place of death:</h2><p>Score: <b>{result["death_place"]["score"]} of {DEATH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein distance: <b>{result["death_place"]["mean_levenshtein_distance"]}<b></p>{table_header}'
        for value_pair in result['forename']['matched_pairs']:
            output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_distance"]}</td></tr>'
        output += '</table>'

    return output


