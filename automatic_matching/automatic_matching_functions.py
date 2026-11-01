import icu
import re
import numpy as np
from difflib import *
import datetime
from dateutil.relativedelta import relativedelta

from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance, damerau_levenshtein_distance_seqs
from rapidfuzz import fuzz
from doublemetaphone import doublemetaphone

AUTOMATIC_MATCHING_ALGORITHM_VERSION_STRING = "2.9"

DATE_COMPARISON_BY_TIMEDELTA_MAX_NUMBER_OF_DAYS = 4
# An imprecise date (a **-wildcard, an open-ended </> bound, or any multi-day
# range) that OVERLAPS the other side is a MODERATE partial match — the same score
# whichever side is 'local'. ~+0.5 on the [-1, 1] date scale (so +10 of the 20
# birth-date points), matching a same-year/different-day agreement: a real full
# date always outweighs a wildcard, and it never flips to a conflict by direction.
DATE_IMPRECISE_PARTIAL_SCORE = 0.5

# -- Corroborating signals --------------------------------------------------
# Distinctive, NON-identity data points (a prisoner number, a former address, an
# imprisonment place) that the six-field identity score never looks at. On a NEAR
# MISS they can tip a pair over the line: positive-only (they only ever ADD), and
# evaluated only when the identity score is already close. Declarative so the set
# is editable without touching the arithmetic; see ``corroborating_boost``.
#
# Each signal:
#   field         key read from each side's data set (mirrors provided_additional_data)
#   compare       'number' (OCR-tolerant near-exact digits) | 'place' (string/phonetic)
#   boost         points added when the field matches on both sides
#   require_match optional guard fields that, IF present on both sides, must ALSO
#                 match for the boost to apply (a mismatch WITHHOLDS the boost — it
#                 is never a penalty). A guard is a bare field name (compared as a
#                 place) or a {'field', 'compare'} dict.
CORROBORATING_SIGNALS = (
    {
        'field': 'prisoner_number',
        'compare': 'number',
        # Tiered: a reused number + matching name but NO date to verify is only a weak
        # tip (+5, won't auto-merge a names-only pair on its own); the full +10 applies
        # only when a birth date ALSO agrees within tolerance (``corroborated_by``).
        'boost': 5,
        'boost_corroborated': 10,
        'corroborated_by': 'birth_date',
        # Prisoner numbers were assigned PER CAMP and reused, so the same number can
        # belong to two people in two camps — require the imprisonment place to agree
        # when both records carry one.
        'require_match': ('place_of_imprisonment',),
        # A reused number only corroborates when the rest of the identity is a small
        # OCR/spelling edit apart (Anna↔Hanna = 1, a one-digit year slip = 1, a month
        # swap 04↔11 = 2) — withhold entirely when the name or date genuinely diverges.
        'require_close': {'surname': 2, 'forename': 2, 'birth_date': 2},
    },
    {
        'field': 'residence_place',
        'compare': 'place',
        'boost': 5,
    },
)
# A place sub-score (cosine space, −1…1) at/above which two places "match".
CORROBORATING_PLACE_MATCH_MIN = 0.5

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


def number_normalization_for_common_ocr_mistakes(value):
    return value.replace("7", "1")

month_day_weights = np.array([
    0,
    0.5,
    0.75,
    0,
    0,
])

year_weights = np.array([
    0,
    0.75,
])

TTP_MATCHING_DEFAULT_DISREGARD_VALUES = {
    'forenames': ['Israel', 'Sarah', 'Sara'],
    'birth_place': ['Deutsches', 'Reich']
}

def get_date_sequences(date):
    normalized_year = number_normalization_for_common_ocr_mistakes(date['year'])
    if abs(int(normalized_year) - int(date['year'])) > 10:
        # with such wide discrepancies we'd hope the error should have been caught. Otherwise a match between 1971 & 1911 would be possible 
        normalized_year = date['year']
        
    normalized_month = number_normalization_for_common_ocr_mistakes(date['month'])
    normalized_day = number_normalization_for_common_ocr_mistakes(date['day'])
    
    month_day_sequences = [
        f"{date['month']}-{date['day']}", # x
        f"{date['day']}-{date['month']}", # 0.5 + 4 * x
        f"{normalized_month}-{normalized_day}", # 0.75 + 4 * x,
        f"{date['month']}-**", # 4 * x
        "**-**", # 4 * x
    ]
    year_sequences = [
        date['year'], # x
        date['year'][:2] + date['year'][3] + date['year'][2] # 0.5 + 2 * x
    ]

    return {
        'month_day_sequences': month_day_sequences,
        'year_sequences': year_sequences,
    }

def convert_dates(dates):
    thresholds = {}
    dates_list = []
    for date_string in dates:
        date_split = date_string.split('-')
        if len(date_split) != 3:
            continue
        if len(date_split[0]) not in [4,5] or len(date_split[1]) != 2 or len(date_split[2]) != 2:
            continue
        date = {
            'year': date_split[0],
            'month': date_split[1],
            'day': date_split[2]
        }

        month = date_split[1]
        day = date_split[2]
        
        
        timedelta_kwargs = {}


        if not month.isdigit():
            month = "01"
            timedelta_kwargs['years'] = 1
        elif len(month) == 1:
            date['month'] = '0' + date['month']
        if not day.isdigit():
            day = "01"
            if 'years' not in timedelta_kwargs:
                timedelta_kwargs['months'] = 1
        else:
            if len(day) == 1:
                date['day'] = '0' + date['day']
            timedelta_kwargs['hours'] = 23
            timedelta_kwargs['minutes'] = 59
        if date['year'][0] == ">":
            date['year'] = date['year'][1:]
            if date['year'].isdigit():
                try:
                    threshold = datetime.datetime(int(date['year']), int(month), int(day), 0)
                    if 'min' not in thresholds or threshold < thresholds['min']:
                        thresholds['min'] = threshold
                except:
                    pass
        elif date['year'][0] == "<":
            date['year'] = date['year'][1:]
            if date['year'].isdigit():
                try:
                    threshold = datetime.datetime(int(date['year']), int(month), int(day), 0) + relativedelta(**timedelta_kwargs)
                    if 'max' not in thresholds or threshold > thresholds['max']:
                        thresholds['max'] = threshold
                except:
                    pass
        elif date['year'].isdigit():
            # String comparisons are only applied to dates that aren't preceded by < or >
            daterange = {}
            try:
                date_from = datetime.datetime(int(date['year']), int(month), int(day), 0)
                date_to = date_from + relativedelta(**timedelta_kwargs)
                daterange = {
                    'datetime_from': date_from,
                    'datetime_to': date_to
                }
                #if 'min' not in date_range or date_from < date_range['min']:
                #    date_range['min'] = date_from
                #if 'max' not in date_range or date_to > date_range['max']:
                #    date_range['max'] = date_to
            except:
                pass
                  
            dates_list.append({
                'date': date,
                **get_date_sequences(date),
                **daterange
            })

    return {
        'thresholds': thresholds,
        'dates': dates_list,
        #'range': date_range
    }

def daterange_as_string(threshold_min=None, threshold_max=None):
    if threshold_min != None and threshold_max != None:
        return f"{threshold_min:%Y-%m-%d} <= X <= {threshold_max:%Y-%m-%d}"
    if threshold_min != None:
        return f"{threshold_min:%Y-%m-%d} <= X"
    if threshold_max != None:
        return f"X <= {threshold_max:%Y-%m-%d}"
    return ' - '

def datetime_range_matches_date(thresholds, dates, external_range=False):

    range_label = 'local'
    date_label = 'external'
    if external_range:
        range_label = 'external'
        date_label = 'local'

    threshold_min = thresholds.get('min', None)
    threshold_max = thresholds.get('max', None)
    matches = []
    non_matches = []
    if threshold_min and threshold_max:
        # Specified date range:                      |===============|
        # Possible scenarios:            |===============| ✓
        #                                                       |======| ✓
        #                                                       | ✓
        #                                              |===========| ✓
        #                                                               |==============| x
        for date in dates:
            if 'datetime_from' not in date:
                continue
            if threshold_min <= date['datetime_to'] and threshold_max >= date['datetime_from']:
                matches.append({
                    range_label: daterange_as_string(threshold_min, threshold_max),
                    date_label: f"{date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
            else:
                matches.append({
                    range_label: daterange_as_string(threshold_min, threshold_max),
                    date_label: f"! {date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
                
    elif threshold_min:
        # Specified date range:                                       | =>
        # Possible scenarios:                    |===============| x
        #                                                         |======| ✓
        #                                                                 | ✓
        #                                                  |===========| ✓
        #                                                               |==============| ✓
        for date in dates:
            if 'datetime_from' not in date:
                continue
            if threshold_min <= date['datetime_to'] or threshold_min <= date['datetime_from']:
                matches.append({
                    range_label: daterange_as_string(threshold_min, None),
                    date_label: f"{date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
            else:
                non_matches.append({
                    range_label: daterange_as_string(threshold_min, None),
                    date_label: f"! {date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
                
    elif threshold_max:
        # Specified date range:                                  => | 
        # Possible scenarios:                    |===============| ✓
        #                                                         |======| ✓
        #                                                                 | ✓
        #                                                  |===========| ✓
        #                                                               |==============| x
        for date in dates:
            if 'datetime_from' not in date:
                continue
            if threshold_max >= date['datetime_to'] or threshold_max >= date['datetime_from']:
                matches.append({
                    range_label: daterange_as_string(None, threshold_max),
                    date_label: f"{date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
            else:
                non_matches.append({
                    range_label: daterange_as_string(None, threshold_max),
                    date_label: f"! {date['year_sequences'][0]}-{date['month_day_sequences'][0]}"
                })
                
            
    return matches, non_matches

def _date_bounds(converted):
    """Each date on one side as an interval ``(lo, hi, precise, label)``. Full and
    ``**``-fuzzy dates come from ``dates`` (their datetime span); open-ended
    ``<``/``>`` come from ``thresholds``. ``precise`` is True only for a single-day
    span (a full date) — everything wider is imprecise."""
    bounds = []
    for d in converted.get('dates', []):
        lo, hi = d.get('datetime_from'), d.get('datetime_to')
        if lo and hi:
            label = f"{d['year_sequences'][0]}-{d['month_day_sequences'][0]}"
            bounds.append((lo, hi, (hi - lo).days <= 1, label))
    thresholds = converted.get('thresholds') or {}
    if thresholds:
        lo = thresholds.get('min', datetime.datetime.min)
        hi = thresholds.get('max', datetime.datetime.max)
        bounds.append((lo, hi, False, daterange_as_string(thresholds.get('min'), thresholds.get('max'))))
    return bounds


def match_date_against_local_date(local_dates, external_dates):
    scores = []
    result = {}
    converted_local_dates = convert_dates(local_dates)
    converted_external_dates = convert_dates(external_dates)

    # Symmetric handling of imprecise dates. When an interval on either side is
    # imprecise (a **-wildcard, a </> bound, or a multi-day range), it is treated as
    # a RANGE — scored the same whichever side is 'local', fixing the old
    # order-dependent scoring that rated a wildcard-vs-full pair a strong match one
    # way and a conflict the other:
    #   · the ranges OVERLAP  → MODERATE partial (the dates are compatible, but the
    #     imprecise side pins nothing down), so a real full date still outscores it;
    #   · the ranges are DISJOINT → conflict (the known parts disagree, e.g. 1925 vs
    #     a 1926 wildcard).
    # Two precise dates are left to the graded scoring below (exact day → strong
    # match; nearby days → partial), so full-date behaviour is unchanged.
    local_bounds = _date_bounds(converted_local_dates)
    external_bounds = _date_bounds(converted_external_dates)
    precise_exact = False
    imprecise_result = None  # best-scoring imprecise pair
    for lo_l, hi_l, prec_l, lbl_l in local_bounds:
        for lo_e, hi_e, prec_e, lbl_e in external_bounds:
            # Upper bounds are EXCLUSIVE (a year wildcard's datetime_to is next-Jan-1,
            # a full date's is 23:59 the same day), so overlap is a STRICT inequality:
            # adjacent ranges that merely touch at the boundary (1912-**-** ending
            # 1913-01-01 vs 1913-**-** starting 1913-01-01) must NOT count as overlap.
            overlaps = lo_l < hi_e and lo_e < hi_l
            if prec_l and prec_e:
                precise_exact = precise_exact or overlaps
                continue
            score = DATE_IMPRECISE_PARTIAL_SCORE if overlaps else -1.0
            if imprecise_result is None or score > imprecise_result['score']:
                imprecise_result = {'local': lbl_l, 'external': lbl_e, 'score': score}
    if not precise_exact and imprecise_result is not None:
        return imprecise_result

    converted_local_dates_thresholds_len = len(converted_local_dates['thresholds'])
    converted_external_dates_thresholds_len = len(converted_external_dates['thresholds'])

    matched_date_ranges = []
    non_matched_date_ranges = []

    if converted_local_dates_thresholds_len > 0 and converted_external_dates_thresholds_len > 0:
        converted_local_dates_min = converted_local_dates['thresholds'].get('min', None)
        converted_local_dates_max = converted_local_dates['thresholds'].get('max', None)
        converted_external_dates_min = converted_external_dates['thresholds'].get('min', None)
        converted_external_dates_max = converted_external_dates['thresholds'].get('max', None)

        compared_date_ranges = False
        if converted_local_dates_min and converted_external_dates_max:
            compared_date_ranges = True
            if converted_local_dates_min > converted_external_dates_max:
                non_matched_date_ranges.append({
                    'local': '!> ' + daterange_as_string(converted_local_dates_min, converted_local_dates_max),
                    'external': '! ' + daterange_as_string(converted_external_dates_min, converted_external_dates_max),
                })

        if converted_local_dates_max and converted_external_dates_min:
            compared_date_ranges = True
            if converted_local_dates_max < converted_external_dates_min:
                non_matched_date_ranges.append({
                    'local': '! ' + daterange_as_string(converted_local_dates_min, converted_local_dates_max),
                    'external': '!> ' + daterange_as_string(converted_external_dates_min, converted_external_dates_max),
                })

        if compared_date_ranges and len(non_matched_date_ranges) == 0:
            matched_date_ranges.append({
                'local': daterange_as_string(converted_local_dates_min, converted_local_dates_max),
                'external': daterange_as_string(converted_external_dates_min, converted_external_dates_max),
            })

    elif converted_local_dates_thresholds_len > 0:
        matches, non_matches = datetime_range_matches_date(converted_local_dates['thresholds'], converted_external_dates['dates'])
        matched_date_ranges += matches
        non_matched_date_ranges += non_matches
    elif converted_external_dates_thresholds_len > 0:
        matches, non_matches = datetime_range_matches_date(converted_external_dates['thresholds'], converted_local_dates['dates'])
        matched_date_ranges += matches
        non_matched_date_ranges += non_matches
            
    if len(matched_date_ranges) > 0:
        external = ', '.join([x['external'] for x in matched_date_ranges])
        local = ', '.join([x['local'] for x in matched_date_ranges])
        result = {
            'external': external,
            'local': local,
            'score': 1
        }
    else:
        for local_date in converted_local_dates['dates']:
            for external_date in converted_external_dates['dates']:
                month_day_comparison = np.array(damerau_levenshtein_distance_seqs(external_date['month_day_sequences'][0], local_date['month_day_sequences']))
                month_day_comparison[1:] *= 6
                month_day_comparison = month_day_comparison + month_day_weights
                year_comparison = np.array(damerau_levenshtein_distance_seqs(external_date['year_sequences'][0], local_date['year_sequences']))
                year_comparison[1:] *= 2
                year_comparison = year_comparison + year_weights
                string_score = np.min(month_day_comparison) + np.min(year_comparison)

                string_score = min(string_score, 3) / 3
                timedelta_score = 1
                timedelta_abs = 0
                if 'datetime_from' in local_date and 'datetime_from' in external_date:
                    timedelta_abs = abs((local_date['datetime_from'] - external_date['datetime_from']).days)
                    timedelta_score = timedelta_abs / DATE_COMPARISON_BY_TIMEDELTA_MAX_NUMBER_OF_DAYS
                    if timedelta_abs > 10:
                        string_score += timedelta_abs / (100 * 356)
                score = min([string_score, timedelta_score, 1])
                scores.append({
                    'external': f"{external_date['year_sequences'][0]}-{external_date['month_day_sequences'][0]}",
                    'local': f"{local_date['year_sequences'][0]}-{local_date['month_day_sequences'][0]}",
                    'score': score,
                    'string_score': string_score,
                    'timedelta_score': timedelta_score,
                })
    
        if len(scores) > 0:
            scores = sorted(scores, key=lambda x: x['score'])
            result = scores[0]
            result['score'] = np.cos(result['score'] * np.pi)

    if len(result) == 0:
        if len(non_matched_date_ranges) > 0:
            external = ', '.join([x['external'] for x in non_matched_date_ranges])
            local = ', '.join([x['local'] for x in non_matched_date_ranges])
            result = {
                'external': external,
                'local': local,
                'score': -1
            }
        else:
            result = {
                'external': "! " + ", ".join(external_dates),
                'local': "! " + ", ".join(local_dates),
                'info': 'Could not compare',
                'score': 0,
            }

    return result

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

def get_doublemetaphone_matching_score(val_1, val_2, potential_shortform = False):
    """
        Returns a range from 0 (perfect match) to 1 (significant differences)
        the flag potential_shortform indicates that names might be shortened, 
        This way we are able to consider that e.g. Alex and Alexander are
        highly likely equivalent as a name.
    """
    dm_val_1 = doublemetaphone(val_1)
    dm_val_2 = doublemetaphone(val_2)

    metaphone_sim = 0
    min_val_len = min(len(val_1), len(val_2))
    if min_val_len > 0:
        min_val_len = min_val_len if min_val_len > 0 else 1

        dm_min_len_1 = max(min(len(dm_val_1[0]), len(dm_val_2[0])), 1)
        dm_min_len_2 = max(min(len(dm_val_1[1]), len(dm_val_2[1])), 1)

        similarity_1 = max(1 - levenshtein_distance(dm_val_1[0], dm_val_2[0]) / dm_min_len_1, 0)
        similarity_2 = max(1 - levenshtein_distance(dm_val_1[1], dm_val_2[1]) / dm_min_len_2, 0)
        
        dlr = max( 1 - damerau_levenshtein_distance(val_1.lower(), val_2.lower()) / min_val_len, 0)
        
        if potential_shortform:
            dm_matching_blocks_1 = SequenceMatcher(None, dm_val_1[0], dm_val_2[0]).get_matching_blocks()
            if dm_matching_blocks_1[0][2] == dm_min_len_1 and dm_matching_blocks_1[1][2] == 0:
                dm_matching_blocks_2 = SequenceMatcher(None, dm_val_1[1], dm_val_2[1]).get_matching_blocks()
                if dm_matching_blocks_2[0][2] == dm_min_len_2 and dm_matching_blocks_2[1][2] == 0:
                    if dm_min_len_1 <= 2 or dm_min_len_2 <= 2:
                        dlr_part = fuzz.partial_ratio(val_1.lower(), val_2.lower())
                        if dlr_part >= 85:
                            similarity_1 = 1
                            similarity_2 = 1 
                    else:
                        similarity_1 = 1
                        similarity_2 = 1
        metaphone_sim = (similarity_1 + similarity_2) / 2
        if metaphone_sim < 1:
            metaphone_sim = (metaphone_sim + dlr) / 2
        else:
            metaphone_sim = (3 * metaphone_sim + dlr) / 4
    
    return 1 - metaphone_sim


def split_string_values(value):
    val = value.replace('(', '').replace(')', '').replace(':', '')
    return list(filter(None, re.split('; |, |/ |-| ', val)))

# Nobiliary / toponymic surname particles (articles + prepositions) across the
# European languages the persecutees came from. Short and ubiquitous, they
# collide phonetically with real name stems (the "Veen" ↔ "Van" false match), so
# they are dropped from the phonetic comparison. Patronymic prefixes (Mac, Mc,
# O', ap, ab, bin, ibn, ben) are deliberately NOT here — they are part of the
# surname, not droppable. Single-character particles (y, e, i, z, d', 't) are
# already removed by the engine's acronym filter.
_SURNAME_PARTICLES_RAW = [
    # German / Austrian / Swiss / Luxembourgish
    'von', 'vom', 'zu', 'zur', 'zum', 'am', 'an', 'im', 'in', 'aus', 'auf', 'unter', 'ob', 'vor', 'zer',
    # Dutch / Flemish / Afrikaans (tussenvoegsels)
    'van', 'ver', 'den', 'der', 'des', 'ten', 'ter', 'te', 'het', 'op', 'aan', 'uit', 'uijt', 'onder',
    'over', 'voor', 'bij', 'tot', 'thoe',
    # French / Walloon
    'de', 'du', 'des', 'le', 'la', 'les',
    # Italian
    'di', 'da', 'del', 'dello', 'della', 'delle', 'dei', 'degli',
    'dal', 'dallo', 'dalla', 'dalle', 'dai', 'dagli', 'lo', 'li',
    # Spanish / Catalan
    'las', 'los', 'dels',
    # Portuguese / Galician
    'do', 'dos', 'das',
    # Scandinavian
    'af', 'av',
    # West-Slavic nobiliary ("of")
    'ze',
]
_SURNAME_PARTICLE_KEYS = frozenset(normalize_string(p, True).lower() for p in _SURNAME_PARTICLES_RAW)


def _demote_surname_particles(result):
    """Drop nobiliary particles from the NORMALISED surname keys (so they no
    longer contribute to the phonetic comparison — "van Klaveren" is matched on
    "Klaveren"), while re-filing their originals under a surviving stem so the
    exact/original layer still sees the full surname and can penalise a
    particle that is present in one record but not the other. Left untouched when
    the surname is *only* particles (nothing to demote onto)."""
    particle_keys = [k for k in result if k.lower() in _SURNAME_PARTICLE_KEYS]
    if not particle_keys:
        return result
    survivors = [k for k in result if k.lower() not in _SURNAME_PARTICLE_KEYS]
    if not survivors:
        return result
    sink = survivors[0]
    for key in particle_keys:
        result[sink].extend(result.pop(key))
    return result


def get_names_as_dict(names, is_surname=False, remove_acronyms=True):
    result = {}
    names_split = []
    for name in names:
        names_split += split_string_values(name)
    for name in names_split:
        name_length = len(name)
        if remove_acronyms:
            if name_length == 1:
                continue
            if remove_acronyms and name_length == 2 and name[-1] == '.':
                continue
        if name_length > 0:
            normalized_value = normalize_string(name, is_surname)
            if normalized_value not in result:
                result[normalized_value] = []
            result[normalized_value].append(name)
    if is_surname:
        result = _demote_surname_particles(result)
    return result


def get_names_as_list_flattend(values, is_surname = False):
    result = []
    dict_values = get_names_as_dict(values, is_surname)
    for val in dict_values:
        result.append(val)
        for orig_val in dict_values[val]:
            result.append(orig_val)
    return ', '.join(result)

def match_against_local_data(local_data, external_data, disregard_data_set={}, potential_shortform = False):
    '''
        Takes two lists of local and external values and compares them.
        Returns a value between -1 (no match) and 1 (perfect match), following a cosine function.
    '''
    matched_pairs = []
    larger_data_set = external_data
    larger_data_set_label = 'external'
    smaller_data_set = local_data
    smaller_data_set_label = 'local'
    score = 0
    if len(local_data) > len(external_data):
        larger_data_set = local_data
        larger_data_set_label = 'local'
        smaller_data_set = external_data
        smaller_data_set_label = 'external'

    names_in_larger_set_normalized = {}
    names_in_larger_set_original = {}
    names_in_smaller_set_normalized = {}
    names_in_smaller_set_original = {}
    names_in_disregard_set_normalized = {}
    names_in_disregard_set_original = {}
    for larger in larger_data_set:
        names_in_larger_set_normalized[larger] = []
        for key in larger_data_set[larger]:
            names_in_larger_set_original[key] = []
    for smaller in smaller_data_set:
        names_in_smaller_set_normalized[smaller] = []
        for key in smaller_data_set[smaller]:
            names_in_smaller_set_original[key] = []

    names_matching_disregard_value_in_smaller_set_normalized = 0
    names_matching_disregard_value_in_smaller_set_original = 0
    names_matching_disregard_value_in_larger_set_normalized = 0
    names_matching_disregard_value_in_larger_set_original = 0
    for disregard in disregard_data_set:
        names_in_disregard_set_normalized[disregard] = []
        if disregard in names_in_smaller_set_normalized:
            names_matching_disregard_value_in_smaller_set_normalized += 1
        if disregard in names_in_larger_set_normalized:
            names_matching_disregard_value_in_larger_set_normalized += 1
        for key in disregard_data_set[disregard]:
            names_in_disregard_set_original[key] = []
            if key in names_in_smaller_set_original:
                names_matching_disregard_value_in_smaller_set_original += 1
            if key in names_in_larger_set_original:
                names_matching_disregard_value_in_larger_set_original += 1
        
    for larger_original in names_in_larger_set_original:
        damerau_levenshtein_distance_sequences = damerau_levenshtein_distance_seqs(larger_original, list(names_in_smaller_set_original.keys()))
        if min(damerau_levenshtein_distance_sequences) == 0:
            # found exact expression in other data set no need for further search.
            names_in_larger_set_original[larger_original] = 0
            if larger_original in names_in_smaller_set_original: # if there is an exact match in the smaller data set we can set this to 0 as well
                names_in_smaller_set_original[larger_original] = 0
        else:
            for smaller_original in smaller_data_set:
                doublemetaphone_matching_score = get_doublemetaphone_matching_score(larger_original, smaller_original, potential_shortform)
                names_in_larger_set_original[larger_original].append(doublemetaphone_matching_score)
                if doublemetaphone_matching_score == 0:
                    break
            names_in_larger_set_original[larger_original] = min(names_in_larger_set_original[larger_original])

    for smaller_original in names_in_smaller_set_original:
        if names_in_smaller_set_original[smaller_original] == 0:
            continue

        for larger_original in larger_data_set:
            doublemetaphone_matching_score = get_doublemetaphone_matching_score(larger_original, smaller_original, potential_shortform)
            names_in_smaller_set_original[smaller_original].append(doublemetaphone_matching_score)
            if doublemetaphone_matching_score == 0:
                break
        
        names_in_smaller_set_original[smaller_original] = min(names_in_smaller_set_original[smaller_original])

    for larger in larger_data_set:
        damerau_levenshtein_distance_sequences = damerau_levenshtein_distance_seqs(larger, list(smaller_data_set.keys()))
        if min(damerau_levenshtein_distance_sequences) == 0:
            # a perfect match has been found for the normalized names.
            names_in_larger_set_normalized[larger] = 0
            if larger in names_in_smaller_set_normalized: # if there is an exact match in the smaller data set we can set this to 0 as well
                names_in_smaller_set_normalized[larger] = 0
        else:
            for smaller in smaller_data_set:
                normalized_doublemetaphone_matching_score = get_doublemetaphone_matching_score(larger, smaller, potential_shortform)
                names_in_larger_set_normalized[larger].append(normalized_doublemetaphone_matching_score)
                if normalized_doublemetaphone_matching_score == 0:
                    # if for the normalized score a perfect match was found we end our search here
                    break

            names_in_larger_set_normalized[larger] = min(names_in_larger_set_normalized[larger])

    
    for smaller in smaller_data_set:
        if names_in_smaller_set_normalized[smaller] == 0:
            # if a perfect match was already found in the previous loop
            continue
        damerau_levenshtein_distance_sequences = damerau_levenshtein_distance_seqs(smaller, list(larger_data_set.keys()))
        if min(damerau_levenshtein_distance_sequences) == 0:
            # a perfect match has been found for the normalized names.
            names_in_smaller_set_normalized[smaller] = 0
        else:
            for larger in larger_data_set:
                normalized_doublemetaphone_matching_score = get_doublemetaphone_matching_score(larger, smaller, potential_shortform)
                names_in_smaller_set_normalized[smaller].append(normalized_doublemetaphone_matching_score)
                if normalized_doublemetaphone_matching_score == 0:
                    # if for the normalized score a perfect match was found we end our search here
                    break
                
            names_in_smaller_set_normalized[smaller] = min(names_in_smaller_set_normalized[smaller])
        
    smaller_data_set_scores = []
    larger_data_set_scores = []
    
    smaller_original_data_set_scores = []
    larger_original_data_set_scores = []


    names_in_disregard_set_normalized_len = len(names_in_disregard_set_normalized)
    names_in_disregard_set_original_len = len(names_in_disregard_set_original)
    
    names_in_smaller_set_normalized_len = len(names_in_smaller_set_normalized)
    for smaller in names_in_smaller_set_normalized:
        if names_in_disregard_set_normalized_len > 0 and names_matching_disregard_value_in_smaller_set_normalized < names_in_smaller_set_normalized_len:
            if smaller in names_in_disregard_set_normalized and names_in_smaller_set_normalized[smaller] > 0.001:
                continue
        smaller_data_set_scores.append(names_in_smaller_set_normalized[smaller])

    names_in_larger_set_normalized_len = len(names_in_larger_set_normalized)
    for larger in names_in_larger_set_normalized:
        if names_in_disregard_set_normalized_len > 0 and names_matching_disregard_value_in_larger_set_normalized < names_in_larger_set_normalized_len:
            if larger in names_in_disregard_set_normalized and names_in_larger_set_normalized[larger] > 0.001:
                continue
        larger_data_set_scores.append(names_in_larger_set_normalized[larger])

    names_in_smaller_set_original_len = len(names_in_smaller_set_original)
    for smaller_original in names_in_smaller_set_original:
        if names_in_disregard_set_original_len > 0 and names_matching_disregard_value_in_smaller_set_original < names_in_smaller_set_original_len:
            if smaller_original in names_in_disregard_set_original and names_in_smaller_set_original[smaller_original] > 0.001:
                continue
        smaller_original_data_set_scores.append(names_in_smaller_set_original[smaller_original])

    names_in_larger_set_original_len = len(names_in_larger_set_original)
    for larger_original in names_in_larger_set_original:
        if names_in_disregard_set_original_len > 0 and names_matching_disregard_value_in_larger_set_original < names_in_larger_set_original_len:
            if larger_original in names_in_disregard_set_original and names_in_larger_set_original[larger_original] > 0.001:
                continue
        larger_original_data_set_scores.append(names_in_larger_set_original[larger_original])

    smaller_data_set_scores.sort()
    larger_data_set_scores.sort()


    smaller_data_set_score = np.cos(np.pi * (8 * np.mean(smaller_data_set_scores) + 4 * max(smaller_data_set_scores) + 2 * np.mean(smaller_original_data_set_scores) + max(smaller_original_data_set_scores)) / 15)
    larger_data_set_score = np.cos(np.pi * (8 * np.mean(larger_data_set_scores) + 4 * max(larger_data_set_scores) + 2 * np.mean(larger_original_data_set_scores) + max(larger_original_data_set_scores)) / 15)

    length_difference_smaller_to_larger_score_set =  len(larger_data_set_scores) - len(smaller_data_set_scores)

    if len(local_data) == len(external_data):
        score = (smaller_data_set_score + larger_data_set_score ) / 2
    else:
        score = (4*smaller_data_set_score + larger_data_set_score ) / 5

    return {
        'score': score,
        'smaller_data_set_score': smaller_data_set_score,
        'larger_data_set_score': larger_data_set_score,
        'difference_in_length': length_difference_smaller_to_larger_score_set,
        f'{smaller_data_set_label}_score': smaller_data_set_score,
        f'{larger_data_set_label}_score': larger_data_set_score,
        smaller_data_set_label: names_in_smaller_set_normalized,
        larger_data_set_label: names_in_larger_set_normalized,
    }



        
    #     temp_levenshtein_ratio_normalized = []
    #     temp_levenshtein_ratio_original = []
    #     for smaller in smaller_data_set:
    #         dmv_normalized = damerau_levenshtein_distance(smaller['normalized'], larger['normalized'])
    #         mean_length_normalized = (len(smaller['normalized']) + len(larger['normalized'])) / 2
    #         lvs_ratio_normalized = np.cos( 2*dmv_normalized / (mean_length_normalized))
    #         lvs_ratio_normalized = lvs_ratio_normalized if lvs_ratio_normalized > 0 else 0
    #         temp_levenshtein_ratio_normalized.append(lvs_ratio_normalized)
    #         dmv_original = damerau_levenshtein_distance(smaller['original'], larger['original'])
    #         mean_length_original = (len(smaller['original']) + len(larger['original'])) / 2
    #         lvs_ratio_original = np.cos( 2*dmv_original / (mean_length_original))
    #         lvs_ratio_original = lvs_ratio_original if lvs_ratio_original > 0 else 0
    #         temp_levenshtein_ratio_original.append(lvs_ratio_original)
    #         matched_pairs.append({
    #             'levenshtein_ratio_normalized': lvs_ratio_normalized,
    #             'levenshtein_ratio_original': lvs_ratio_original,
    #             larger_data_set_label: larger,
    #             smaller_data_set_label: smaller
    #         })
    #         if lvs_ratio_normalized == 1:
    #             # A perfect fit has been found, no further matching is necessary
    #             break
    #     # As some of the names might occur more than once especially after normalization
    #     names_in_larger_set_normalized[larger['normalized']].append(max(temp_levenshtein_ratio_normalized))
    #     names_in_larger_set_original[larger['original']].append(max(temp_levenshtein_ratio_original))
    #     names_in_smaller_set_normalized[smaller['normalized']].append(max(temp_levenshtein_ratio_normalized))
    #     names_in_smaller_set_original[smaller['original']].append(max(temp_levenshtein_ratio_original))
        
    # levenshtein_ratios_smaller_set_normalized = []
    # levenshtein_ratios_smaller_set_orginal = []
    # levenshtein_ratios_larger_set_normalized = []
    # levenshtein_ratios_larger_set_orginal = []

    # for smaller_name_normalized in names_in_smaller_set_normalized:
    #     tmp_val = 0
    #     if len(names_in_smaller_set_normalized[smaller_name_normalized]) > 0:
    #         tmp_val = np.mean(names_in_smaller_set_normalized[smaller_name_normalized])
    #         if max(names_in_smaller_set_normalized[smaller_name_normalized]) == 1:
    #             tmp_val = 1
    #     levenshtein_ratios_smaller_set_normalized.append(tmp_val)
    # for smaller_name_original in names_in_smaller_set_original:
    #     tmp_val = 0
    #     if len(names_in_smaller_set_original[smaller_name_original]) > 0:
    #         tmp_val = np.mean(names_in_smaller_set_original[smaller_name_original])
    #         if max(names_in_smaller_set_normalized[smaller_name_normalized]) == 1:
    #             tmp_val = 1
    #     levenshtein_ratios_smaller_set_orginal.append(tmp_val)
    # for larger_name_normalized in names_in_larger_set_normalized:
    #     tmp_val = 0
    #     if len(names_in_larger_set_normalized[larger_name_normalized]) > 0:
    #         tmp_val = np.mean(names_in_larger_set_normalized[larger_name_normalized])
    #         if max(names_in_larger_set_normalized[larger_name_normalized]) == 1:
    #             tmp_val = 1
    #     levenshtein_ratios_larger_set_normalized.append(tmp_val)
    # for larger_name_original in names_in_larger_set_original:
    #     tmp_val = 0
    #     if len(names_in_larger_set_original[larger_name_original]) > 0:
    #         tmp_val = np.mean(names_in_larger_set_original[larger_name_original])
    #         if max(names_in_larger_set_normalized[larger_name_normalized]) == 1:
    #             tmp_val = 1
    #     levenshtein_ratios_larger_set_orginal.append(tmp_val)
        


    # matched_pairs_sorted = sorted(matched_pairs, key=lambda x: -x['levenshtein_ratio_normalized'])
    # mean_levenshtein_ratio_normalized = (np.mean(levenshtein_ratios_smaller_set_normalized) + np.mean(levenshtein_ratios_larger_set_normalized)) / 2
    # median_levenshtein_ratio_normalized = (np.median(levenshtein_ratios_smaller_set_normalized) + np.median(levenshtein_ratios_larger_set_normalized) ) / 2
    # mean_levenshtein_ratio_original = (np.mean(levenshtein_ratios_smaller_set_orginal) + np.mean(levenshtein_ratios_larger_set_orginal)) / 2
    # return {
    #     'mean_levenshtein_ratio_normalized': mean_levenshtein_ratio_normalized,
    #     'median_levenshtein_ratio_normalized': median_levenshtein_ratio_normalized,
    #     'mean_levenshtein_ratio_original': mean_levenshtein_ratio_original,
    #     'levenshtein_ratio_normalized': f'Mean: {mean_levenshtein_ratio_normalized}',
    #     'levenshtein_ratio_original': f'Mean: {mean_levenshtein_ratio_original}',
    #     'matched_pairs': matched_pairs_sorted,
    #     'score': (mean_levenshtein_ratio_normalized + median_levenshtein_ratio_normalized) / 2
    # }


FORENAME_MAX_SCORE_CONTRIBUTION = 25
SURNAME_MAX_SCORE_CONTRIBUTION = 25

BIRTH_PLACE_MAX_SCORE_CONTRIBUTION = 10
BIRTH_DATE_MAX_SCORE_CONTRIBUTION = 20

DEATH_PLACE_MAX_SCORE_CONTRIBUTION = 10
DEATH_DATE_MAX_SCORE_CONTRIBUTION = 10

MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING = 60
MIN_TOTAL_SCORE_FOR_MATCH_WITH_PERFECT_RELATIVE_SCORE = 50


TOTAL_MAX_SCORE_REACHABLE = FORENAME_MAX_SCORE_CONTRIBUTION + SURNAME_MAX_SCORE_CONTRIBUTION + BIRTH_PLACE_MAX_SCORE_CONTRIBUTION + BIRTH_DATE_MAX_SCORE_CONTRIBUTION + DEATH_PLACE_MAX_SCORE_CONTRIBUTION + DEATH_DATE_MAX_SCORE_CONTRIBUTION


def get_matching_score(local_data_set, external_data_set, values_to_be_disregarded={}, *,
                        with_edit_distances=False):
    '''
        Expected inputs: local_data_set and external_data_set:
        To get a complete match all values have to be provided.
        Expected layout e.g.:
        {
            'forenames': ['Anna', 'Anne'],
            'surnames': ['Musterfrau', 'Levy'], # (Include all surnames and birthnames, etc. here)
            'birth_place': ['München', 'Bayern'],
            'birth_date': ['YYYY-MM-DD', '<YYYY-MM-DD'], # Use < or > to indicate smaller or greater dates (only the year will be taken into account then) for fuzzy dates use ** instead of MM or DD
            'death_place': ['Dachau'],
            'death_date': ['YYYY-MM-DD'],
        }
    '''
    results = {}
    absolute_score = 0
    absolute_score_original = 0
    max_score_reachable = 0
    max_total_score_reachable = 0
    name_min_score_reached = True
    birth_min_score_reached = True
    death_min_score_reached = True

    # NAME information
    local_forenames = {}
    external_forenames = {}
    disregard_forenames = {}
    forename_results = {}
    if 'forenames' in local_data_set:
        local_forenames = get_names_as_dict(local_data_set['forenames'], False)
        forename_results['local'] = local_forenames

    if 'forenames' in external_data_set:
        external_forenames = get_names_as_dict(external_data_set['forenames'], False)
        forename_results['external'] = external_forenames

    if 'forenames' in values_to_be_disregarded:
        disregard_forenames = get_names_as_dict(values_to_be_disregarded['forenames'], False)
        forename_results['disregard'] = disregard_forenames

    if len(local_forenames) > 0 and len(external_forenames) > 0:
        forename_results = match_against_local_data(local_forenames, external_forenames, disregard_forenames, True)

        forename_score = forename_results['score'] * FORENAME_MAX_SCORE_CONTRIBUTION
        max_score_reachable += FORENAME_MAX_SCORE_CONTRIBUTION
        absolute_score += forename_score

        forename_results['absolute_score'] = forename_score
        forename_results['max_absolute_score'] = FORENAME_MAX_SCORE_CONTRIBUTION

    if len(local_forenames) > 0 or len(external_forenames) > 0:
        max_total_score_reachable += FORENAME_MAX_SCORE_CONTRIBUTION

    results['forename'] = forename_results

    local_surnames = {}
    external_surnames = {}
    disregard_surnames = {}
    surname_results = {}
    if 'surnames' in local_data_set:
        local_surnames = get_names_as_dict(local_data_set['surnames'], True)
        surname_results['local'] = local_surnames

    if 'surnames' in external_data_set:
        external_surnames = get_names_as_dict(external_data_set['surnames'], True)
        surname_results['external'] = external_surnames

    if 'surnames' in values_to_be_disregarded:
        disregard_surnames = get_names_as_dict(values_to_be_disregarded['surnames'], True)
        surname_results['disregard'] = disregard_surnames

    if len(local_surnames) > 0 and len(external_surnames) > 0:
        surname_results = match_against_local_data(local_surnames, external_surnames, disregard_surnames, False)

        surname_score = surname_results['score'] * SURNAME_MAX_SCORE_CONTRIBUTION
        max_score_reachable += SURNAME_MAX_SCORE_CONTRIBUTION
        absolute_score += surname_score

        surname_results['absolute_score'] = surname_score
        surname_results['max_absolute_score'] = SURNAME_MAX_SCORE_CONTRIBUTION

    if len(local_surnames) > 0 or len(external_surnames) > 0:
        max_total_score_reachable += SURNAME_MAX_SCORE_CONTRIBUTION

    results['surname'] = surname_results

    # BIRTH information
            
    
    local_birth_place = {}
    external_birth_place = {}
    disregard_birth_place = {}
    birth_place_results = {}
    if 'birth_place' in local_data_set:
        local_birth_place = get_names_as_dict(local_data_set['birth_place'], False)
        birth_place_results['local'] = local_birth_place

    if 'birth_place' in external_data_set:
        external_birth_place = get_names_as_dict(external_data_set['birth_place'], False)
        birth_place_results['external'] = external_birth_place

    if 'birth_place' in values_to_be_disregarded:
        disregard_birth_place = get_names_as_dict(values_to_be_disregarded['birth_place'], False)
        birth_place_results['disregard'] = disregard_birth_place

    if len(local_birth_place) > 0 and len(external_birth_place) > 0:
        birth_place_results = match_against_local_data(local_birth_place, external_birth_place, disregard_birth_place, True)

        birth_place_results['score'] = birth_place_results['smaller_data_set_score'] # Override the combined score with the one for the smaller data set only (As the place formating might differ to a great extend)

        birth_place_score = birth_place_results['score'] * BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
        max_score_reachable += BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
        absolute_score += birth_place_score

        birth_place_results['absolute_score'] = birth_place_score
        birth_place_results['max_absolute_score'] = BIRTH_PLACE_MAX_SCORE_CONTRIBUTION

    if len(local_birth_place) > 0 or len(external_birth_place) > 0:
        max_total_score_reachable += BIRTH_PLACE_MAX_SCORE_CONTRIBUTION

    results['birth_place'] = birth_place_results


    local_birth_date = {}
    external_birth_date = {}
    birth_date_results = {}
    if 'birth_date' in local_data_set:
        local_birth_date = local_data_set['birth_date']
        birth_date_results['local'] = ', '.join(local_data_set['birth_date'])

    if 'birth_date' in external_data_set:
        external_birth_date = external_data_set['birth_date']
        birth_date_results['external'] = ', '.join(external_data_set['birth_date'])

    if len(local_birth_date) > 0 and len(external_birth_date) > 0:

        birth_date_results = match_date_against_local_date(local_birth_date, external_birth_date)
    
        birth_date_score = BIRTH_DATE_MAX_SCORE_CONTRIBUTION * birth_date_results['score']
        max_score_reachable += BIRTH_DATE_MAX_SCORE_CONTRIBUTION
        absolute_score += birth_date_score

        birth_date_results['absolute_score'] = birth_date_score
        birth_date_results['max_absolute_score'] = BIRTH_DATE_MAX_SCORE_CONTRIBUTION

    if len(local_birth_date) > 0 or len(external_birth_date) > 0:
        max_total_score_reachable += BIRTH_DATE_MAX_SCORE_CONTRIBUTION

    results['birth_date'] = birth_date_results
    # DEATH information
    

    local_death_place = {}
    external_death_place = {}
    disregard_death_place = {}
    death_place_results = {}
    if 'death_place' in local_data_set:
        local_death_place = get_names_as_dict(local_data_set['death_place'], False)
        death_place_results['local'] = local_death_place

    if 'death_place' in external_data_set:
        external_death_place = get_names_as_dict(external_data_set['death_place'], False)
        death_place_results['external'] = external_death_place

    if 'death_place' in values_to_be_disregarded:
        disregard_death_place = get_names_as_dict(values_to_be_disregarded['death_place'], False)
        death_place_results['disregard'] = disregard_death_place

    if len(local_death_place) > 0 and len(external_death_place) > 0:

        death_place_results = match_against_local_data(local_death_place, external_death_place, disregard_death_place, True)

        death_place_results['score'] = death_place_results['smaller_data_set_score']
        death_place_score = death_place_results['score'] * DEATH_PLACE_MAX_SCORE_CONTRIBUTION
        max_score_reachable += DEATH_PLACE_MAX_SCORE_CONTRIBUTION
        absolute_score += death_place_score

        death_place_results['absolute_score'] = death_place_score
        death_place_results['max_absolute_score'] = DEATH_PLACE_MAX_SCORE_CONTRIBUTION

    if len(local_death_place) > 0 or len(external_death_place) > 0:
        max_total_score_reachable += DEATH_PLACE_MAX_SCORE_CONTRIBUTION

    results['death_place'] = death_place_results

    local_death_date = {}
    external_death_date = {}
    death_date_results = {}
    if 'death_date' in local_data_set:
        local_death_date = local_data_set['death_date']
        death_date_results['local'] = ', '.join(local_data_set['death_date'])

    if 'death_date' in external_data_set:
        external_death_date = external_data_set['death_date']
        death_date_results['external'] = ', '.join(external_data_set['death_date'])

    if len(local_death_date) > 0 and len(external_death_date) > 0:

        death_date_results = match_date_against_local_date(local_death_date, external_death_date)
    
        death_date_score = DEATH_DATE_MAX_SCORE_CONTRIBUTION * death_date_results['score']
        max_score_reachable += DEATH_DATE_MAX_SCORE_CONTRIBUTION
        absolute_score += death_date_score

        death_date_results['absolute_score'] = death_date_score
        death_date_results['max_absolute_score'] = DEATH_DATE_MAX_SCORE_CONTRIBUTION
    
    if len(local_death_date) > 0 or len(external_death_date) > 0:
        max_total_score_reachable += DEATH_DATE_MAX_SCORE_CONTRIBUTION

    results['death_date'] = death_date_results

    relative_score = ( absolute_score / max_score_reachable ) if max_score_reachable > 0 else 0
    total_relative_score = ( absolute_score / max_total_score_reachable ) if max_total_score_reachable > 0 else 0

    automatically_matched = absolute_score >= MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING
    if total_relative_score == 1 and absolute_score >= MIN_TOTAL_SCORE_FOR_MATCH_WITH_PERFECT_RELATIVE_SCORE:
        automatically_matched = True
    
    result = {
        **results,
        'absolute_score': absolute_score,
        'relative_score': relative_score,
        'total_relative_score': total_relative_score,
        'max_score_reachable': max_score_reachable,
        'automatically_matched': bool(automatically_matched),
        'matching_algorithm_version': AUTOMATIC_MATCHING_ALGORITHM_VERSION_STRING,
    }
    # The edit-distance readout re-runs name normalisation (icu), so it is opt-in: the
    # hot scoring path leaves it off; the staff explanation (and the near-miss gate, via
    # corroborating_boost) turn it on only where consumed. See docs/corroborating-signals.md.
    if with_edit_distances:
        result['edit_distances'] = _edit_distances(local_data_set, external_data_set)
    return result


# -- edit-distance readouts -------------------------------------------------
# How far apart the two records' names / dates are AFTER the engine's own
# normalisation — the residual spelling/OCR edits. Surfaced on every result so a
# reviewer (and the corroborating-signal gate) can see whether a near miss is "one
# typo away" (Anna↔Hanna = 1; a one-digit year slip 1914↔1904 = 1; a month swap
# 04↔11 = 2) or genuinely different.

def _edit_name_tokens(values, is_surname=False):
    tokens = []
    for value in values or []:
        for token in split_string_values(value):
            norm = normalize_string(token, is_surname)
            if norm:
                tokens.append(norm)
    return tokens


def name_edit_distance(local_values, external_values, is_surname=False):
    """Smallest Damerau–Levenshtein distance between any local and external name
    token, on the engine-normalised forms (so it counts the spelling edits that
    remain once transliteration/folding is applied). ``None`` when either side has
    no name token."""
    local = _edit_name_tokens(local_values, is_surname)
    external = _edit_name_tokens(external_values, is_surname)
    if not local or not external:
        return None
    return min(damerau_levenshtein_distance(a, b) for a in local for b in external)


def _date_digit_cells(date):
    """A date string ('YYYY-MM-DD', possibly with ** wildcards or a </> bound) as 8
    cells (YYYYMMDD); each cell is its digit, or ``None`` when unknown/wildcard. For
    an open bound (< or >) only the year is meaningful — the month/day are sentinels,
    so they are treated as unknown."""
    raw = str(date or '').strip()
    open_bound = raw[:1] in ('<', '>')
    text = raw.lstrip('<>').strip()
    cells = []
    for i, (part, width) in enumerate(zip(text.split('-'), (4, 2, 2))):
        chars = list(part)[:width]
        chars += ['?'] * (width - len(chars))
        for ch in chars:
            cells.append(None if (open_bound and i > 0) else (ch if ch.isdigit() else None))
    cells += [None] * (8 - len(cells))
    return cells[:8]


def _edit_distances(local_data_set, external_data_set):
    """The full readout: name (forename/surname) + birth-date edit distances."""
    return {
        'forename': name_edit_distance(local_data_set.get('forenames'), external_data_set.get('forenames'), False),
        'surname': name_edit_distance(local_data_set.get('surnames'), external_data_set.get('surnames'), True),
        'birth_date': date_edit_distance(local_data_set.get('birth_date'), external_data_set.get('birth_date')),
    }


def date_edit_distance(local_dates, external_dates):
    """Fewest differing digits between the closest pair of dates, over positions
    known on BOTH sides (wildcards / open bounds don't count). A one-digit year slip
    (1914↔1904) = 1, a month swap (04↔11) = 2, a wholly different date is large.
    ``None`` when a date is missing on either side or nothing is comparable."""
    if not local_dates or not external_dates:
        return None
    best = None
    for da in local_dates:
        ca = _date_digit_cells(da)
        for db in external_dates:
            cb = _date_digit_cells(db)
            pairs = [(x, y) for x, y in zip(ca, cb) if x is not None and y is not None]
            if not pairs:
                continue
            diffs = sum(1 for x, y in pairs if x != y)
            best = diffs if best is None else min(best, diffs)
    return best


def corroborating_signal_fields(signals=CORROBORATING_SIGNALS):
    """Every data-set key the signals read (signal fields + guard fields), in order,
    de-duplicated — the set the data-set builders and ES routine must carry."""
    keys = []
    for signal in signals or ():
        candidates = [signal.get('field')]
        for guard in signal.get('require_match', ()):
            candidates.append(guard if isinstance(guard, str) else guard.get('field'))
        for key in candidates:
            if key and key not in keys:
                keys.append(key)
    return tuple(keys)


def _corroborating_values(data_set, field):
    """Normalise a data-set entry to a list of non-empty strings (the fields may be
    scalars from ``provided_additional_data`` or already lists)."""
    value = data_set.get(field)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _corroborating_number_signature(value):
    """Digits only, with the engine's common OCR digit folding applied, so OCR
    confusions (notably 7↔1) don't split an otherwise-identical number. Empty when
    the value holds no digits."""
    digits = re.sub(r'\D', '', str(value or ''))
    return number_normalization_for_common_ocr_mistakes(digits) if digits else ''


def _corroborating_place_similarity(local_value, external_value):
    """The engine's own place similarity (−1…1) between two single place strings."""
    local = get_names_as_dict([local_value], False)
    external = get_names_as_dict([external_value], False)
    if not local or not external:
        return -1.0
    result = match_against_local_data(local, external, {}, True)
    return result.get('smaller_data_set_score', result.get('score', -1.0))


def _corroborating_field_match(compare, local_values, external_values, place_match_min):
    """``(present_on_both, matched)`` for one field. ``present_on_both`` is True only
    when each side carries a value; ``matched`` says whether any pair agrees."""
    if not local_values or not external_values:
        return (False, False)
    if compare == 'number':
        signatures = {_corroborating_number_signature(v) for v in local_values}
        signatures.discard('')
        matched = any(
            _corroborating_number_signature(v) in signatures
            for v in external_values if _corroborating_number_signature(v)
        )
        return (True, matched)
    # place-style
    best = max(
        (_corroborating_place_similarity(a, b) for a in local_values for b in external_values),
        default=-1.0,
    )
    return (True, best >= place_match_min)


def corroborating_boost(result, local_data_set, external_data_set, *,
                        near_miss_min, near_miss_max,
                        signals=CORROBORATING_SIGNALS,
                        place_match_min=CORROBORATING_PLACE_MATCH_MIN,
                        cap=None):
    """Positive-only corroboration, applied ONLY when ``result``'s ``absolute_score``
    is a near miss (``near_miss_min`` <= score < ``near_miss_max``).

    For each configured signal whose field matches on both records — and whose
    ``require_match`` guards all pass and ``require_close`` edit-distance limits hold —
    the signal's ``boost`` is added to the score. A guard present on both sides that
    DISAGREES withholds that signal's boost (never a penalty); a guard missing on
    either side is inconclusive and does not block. ``require_close`` {field: max}
    additionally withholds the boost when a residual name/date edit distance exceeds
    its limit (so a reused identifier can't merge genuinely different identities).
    The applied boost and a per-signal breakdown are recorded on ``result``
    (``corroborating_boost`` / ``corroborating_signals``) so a rescued match stays
    explainable. Outside the near-miss band, or with nothing matching, ``result`` is
    returned unchanged.

    This is an SR-level adjustment layered on the engine's identity arithmetic — like
    the forename-variant credit and the confirmed-individual birth-date penalty — kept
    here so it ships with the published algorithm. The near-miss band must be judged
    AFTER those adjustments, so the caller (not the engine's own scorer) supplies it.
    """
    score = result.get('absolute_score', 0) or 0
    if not (near_miss_min <= score < near_miss_max) or not signals:
        return result

    def edits():
        # Just-in-time, memoised. Reuses the readout if the caller already asked for it
        # (with_edit_distances=True); otherwise computes once, here on a near-miss pair
        # that reached the gate — the only place the hot path needs edit distances.
        e = result.get('edit_distances')
        if e is None:
            e = _edit_distances(local_data_set, external_data_set)
            result['edit_distances'] = e
        return e

    applied = []
    total = 0.0
    for signal in signals:
        field = signal['field']
        boost = signal.get('boost', 0) or 0
        if boost <= 0:
            continue
        present, matched = _corroborating_field_match(
            signal.get('compare', 'place'),
            _corroborating_values(local_data_set, field),
            _corroborating_values(external_data_set, field),
            place_match_min,
        )
        if not (present and matched):
            continue

        guards = {}
        withheld = False
        for guard in signal.get('require_match', ()):
            g_field = guard if isinstance(guard, str) else guard.get('field')
            g_compare = 'place' if isinstance(guard, str) else guard.get('compare', 'place')
            g_present, g_matched = _corroborating_field_match(
                g_compare,
                _corroborating_values(local_data_set, g_field),
                _corroborating_values(external_data_set, g_field),
                place_match_min,
            )
            if not g_present:
                guards[g_field] = 'absent'
            elif g_matched:
                guards[g_field] = 'match'
            else:
                guards[g_field] = 'conflict'
                withheld = True
        if withheld:
            continue

        # Edit-distance gate: an identifier (e.g. a reused prisoner number) only
        # corroborates when the rest of the identity is a small spelling/OCR edit
        # apart — not when the names/dates genuinely diverge. For each field named in
        # ``require_close`` with a readout present (both sides had a value), the edit
        # distance must be within the limit, else the boost is withheld. A missing
        # readout (field absent on a side) is inconclusive and does not block.
        close = {}
        too_far = False
        for cfield, limit in (signal.get('require_close') or {}).items():
            dist = edits().get(cfield)
            if dist is None:
                continue
            close[cfield] = dist
            if dist > limit:
                too_far = True
        if too_far:
            continue

        # Tiered boost: a reused identifier on its own (names only) is a weak tip; the
        # full ``boost_corroborated`` applies only when an independent identity field
        # (``corroborated_by``, e.g. birth_date) is ALSO present on both sides — and,
        # having passed ``require_close`` above, is therefore within tolerance. Absent
        # that corroboration the base ``boost`` applies.
        amount = boost
        corroborated_by = signal.get('corroborated_by')
        if (corroborated_by and signal.get('boost_corroborated')
                and edits().get(corroborated_by) is not None):
            amount = signal['boost_corroborated']

        total += amount
        applied.append({'field': field, 'boost': amount, 'guards': guards,
                        'edits': close, 'corroborated': amount != boost})

    if total <= 0:
        return result
    if cap is not None:
        total = min(total, cap)

    result['absolute_score'] = score + total
    result['corroborating_boost'] = total
    result['corroborating_signals'] = applied
    max_reachable = result.get('max_score_reachable', 0)
    result['relative_score'] = (result['absolute_score'] / max_reachable) if max_reachable else result.get('relative_score', 0)
    return result


def convert_dict_to_string(values, total_score=None):
    result = ""
    for x in values:
        if len(result) > 0:
            result += ', '
        if values[x] != None:
            if type(values[x]) == list:
                result += f'{x} {{{", ".join(values[x])}}}'
            elif type(values[x]) == float:
                result += f'{x} ({values[x]:.2f})'
            else:
                result += f'{x} ({values[x]})'
        else:
            result += x

    if total_score != None:
        result += f' [{total_score:.2f}]'
    return result

def comparison_html_bar_chart(max_absolute_score, score=None, absolute_score=None, relative_score=None):
    red, green, blue = 50, 50, 50
    percentage = 1
    text_anchor = "end"
    text_start = "-4"
    if relative_score != None:
        score = relative_score
    if score != None:
        percentage = (1+score)/2
        blue = 0
        green = 200 * np.sqrt( np.sin ( percentage * np.pi / 2 ))
        red = 255 * np.sqrt( np.cos ( percentage * np.pi / 2 ))
    
    result = f'''<div style="white-space:nowrap; border-top: 2px solid #ccc; background-color: #888; text-align: center; overflow: hidden;">
        <svg xmlns="http://www.w3.org/2000/svg" style="width: 100%; min-height: 16px; margin-bottom: -3px;" viewBox="-100 0 200 30" >'''

    if relative_score == None:
        result += '<line x1="0" y1="0" x2="0" y2="30" stroke-width="2" stroke="#000"/>'
        if score:
            if score < 0:
                text_anchor = "start"
                text_start = "4"
            result += f'<line x1="{score * 100}" y1="15" x2="0" y2="15" stroke-width="30" stroke="rgb({red}, {green}, {blue})"/>'

        absolute_score = f"{absolute_score:.2f}" if absolute_score != None else '-'
        result += f'<text text-anchor="{text_anchor}" x="{text_start}" y="21" font-size="16" font-weight="bold"> {absolute_score} / {max_absolute_score} </text>'
    else:    
        absolute_score = f"{absolute_score:.2f}" if absolute_score != None else '-'
        result += f'<line x1="-100" y1="15" x2="{-100 + relative_score * 200}" y2="15" stroke-width="30" stroke="rgb({red}, {green}, {blue})"/>'
        result += f'<text text-anchor="middle" x="0" y="21" font-size="16" font-weight="bold">{absolute_score} / {max_absolute_score} ({100*relative_score:.2f} %)</text>'

    
    
    
    result += '</svg></div>'

    return result


def comparison_html_table_cell(max_absolute_score, score=None, absolute_score=None, local_value=None, external_value=None, info=None, relative_score=None, automatically_matched=False):
    color = "#fff"
    background = "#555"
    if score != None:
        color = "#000"
        background = "transparent"
    result = f'<td class="comparison" style="background-color: {background}; color: {color}; min-width: 120px; max-width: 140px; vertical-align: bottom;"><div style="display: flex; flex-wrap: nowrap; flex-direction: column; align-items: stretch; border-radius: 4px; overflow: hidden; border: 1px solid #ccc;">'
    if info != None:
        result += f'<div style="padding: 4px 6px;">{info}</div>'
    elif relative_score != None:
        matched = "✅︎" if automatically_matched else "❌"
        result += f'<div style="padding: 4px 6px; text-align: center;">Meets criteria: {matched}</div>'
        result += comparison_html_bar_chart(max_absolute_score, absolute_score=absolute_score, relative_score=relative_score)
    elif external_value != None and local_value != None:
        result += f'<div style="padding: 4px 6px;">{external_value if len(external_value) else "-"}</div><div style="background-color: rgba(0, 169, 176, 0.3); padding: 4px 6px;">{local_value if len(local_value) > 0 else "-"}</div>'
    else:
        result += f'<div style="padding: 4px 6px;">---</div><div style="background-color: rgba(0, 169, 176, 0.3); padding: 4px 6px;">---</div>'
    
    result += comparison_html_bar_chart(max_absolute_score, score, absolute_score) + '</div></td>'
    return result

def get_result_as_html_table_row(potential_match, num=None, **kwargs):
    result = '<tr>'
    if num != None:
        result += f'<td>{num}</td>'
    if 'matched' in kwargs:
        result += f'<td>{kwargs.get("matched")}</td>'

    surname = potential_match.get('surname', {})
    forename = potential_match.get('forename', {})
    birth_date = potential_match.get('birth_date', {})
    birth_place = potential_match.get('birth_place', {})
    death_date = potential_match.get('death_date', {})
    death_place = potential_match.get('death_place', {})

    result += comparison_html_table_cell(potential_match.get('max_score_reachable', 0), score=potential_match.get('absolute_score', 0) / 100, absolute_score=potential_match.get('absolute_score', None), relative_score=potential_match.get('relative_score', None), automatically_matched=potential_match.get('automatically_matched', False))

    result += comparison_html_table_cell(SURNAME_MAX_SCORE_CONTRIBUTION, local_value=convert_dict_to_string(surname.get('local', {}), surname.get('local_score', None)), external_value=convert_dict_to_string(surname.get('external', {}), surname.get('external_score', None)), score=surname.get('score', None), absolute_score=surname.get('absolute_score', None))
    result += comparison_html_table_cell(FORENAME_MAX_SCORE_CONTRIBUTION, local_value=convert_dict_to_string(forename.get('local', {}), forename.get('local_score', None)), external_value=convert_dict_to_string(forename.get('external', {}), forename.get('external_score', None)), score=forename.get('score', None), absolute_score=forename.get('absolute_score', None))

    result += comparison_html_table_cell(BIRTH_DATE_MAX_SCORE_CONTRIBUTION, local_value=birth_date.get('local', ''), external_value=birth_date.get('external', ''), score=birth_date.get('score', None), absolute_score=birth_date.get('absolute_score', None))
    result += comparison_html_table_cell(BIRTH_PLACE_MAX_SCORE_CONTRIBUTION, local_value=convert_dict_to_string(birth_place.get('local', {}), birth_place.get('local_score', None)), external_value=convert_dict_to_string(birth_place.get('external', {}), birth_place.get('external_score', None)), score=birth_place.get('score', None), absolute_score=birth_place.get('absolute_score', None))

    result += comparison_html_table_cell(DEATH_DATE_MAX_SCORE_CONTRIBUTION, local_value=death_date.get('local', ''), external_value=death_date.get('external', ''), score=death_date.get('score', None), absolute_score=death_date.get('absolute_score', None))
    result += comparison_html_table_cell(DEATH_PLACE_MAX_SCORE_CONTRIBUTION, local_value=convert_dict_to_string(death_place.get('local', {}), death_place.get('local_score', None)), external_value=convert_dict_to_string(death_place.get('external', {}), death_place.get('external_score', None)), score=death_place.get('score', None), absolute_score=death_place.get('absolute_score', None))


    if 'es_score' in potential_match:
        result += f'<td>{potential_match.get("es_score")}</td>'
    if 'link' in kwargs:
        result += f'<td>{kwargs.get("link")}</td>'

    result += "</tr>"
    
    return result



def get_results_as_html(results):
    if type(results) != list:
        results = [results]
    output = f'<table><tr style="text-align: center;"><th style="text-align: center;">Score</th><th style="text-align: center;">Surname</th><th style="text-align: center;">Forename</th><th style="text-align: center;">Birth date</th><th style="text-align: center;">Birth place</th><th style="text-align: center;">Death date</th><th style="text-align: center;">Death place</th></tr>'
    for result in results:
        output += get_result_as_html_table_row(result)
    output += '</body></table>'

    # table_header = '<table><tr><th>Type</th><th>Local value</th><th>External value</th><th>Score</th></tr>'
    # if 'forename' in result:
    #     output += f'<tr><td><h2>Forename:</h2></td><td>{", ".join()}</td><td>{result["forename"]["absolute_score"]} of {FORENAME_MAX_SCORE_CONTRIBUTION}</td><td></p><p>Mean Levenshtein ratio: <b>{result["forename"]["mean_levenshtein_ratio_normalized"]} ({result["forename"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
    #     for value_pair in result['forename']['matched_pairs']:
    #         output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
    #     output += '</table>'
        
    # if 'surname' in result:
    #     output += f'<h2>Surname:</h2><p>Score: <b>{result["surname"]["absolute_score"]} of {SURNAME_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["surname"]["mean_levenshtein_ratio_normalized"]} ({result["surname"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
    #     for value_pair in result['surname']['matched_pairs']:
    #         output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
    #     output += '</table>'
    
    # if 'birth_date' in result:
    #     output += f'<h2>Date of birth:</h2><p>Score: <b>{result["birth_date"]["absolute_score"]} of {BIRTH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["birth_date"]["levenshtein_distance"]}<b></p>{table_header}'
    #     output += f'<tr><td>{result["birth_date"]["local"]}</td><td>{result["birth_date"]["external"]}</td><td>{result["birth_date"]["levenshtein_distance"]}</td></tr>'
    #     output += '</table>'

    # if 'birth_place' in result:
    #     output += f'<h2>Place of birth:</h2><p>Score: <b>{result["birth_place"]["absolute_score"]} of {BIRTH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["birth_place"]["mean_levenshtein_ratio_normalized"]} ({result["birth_place"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
    #     for value_pair in result['birth_place']['matched_pairs']:
    #         output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
    #     output += '</table>'

    # if 'death_date' in result:
    #     output += f'<h2>Date of death:</h2><p>Score: <b>{result["death_date"]["absolute_score"]} of {DEATH_DATE_MAX_SCORE_CONTRIBUTION}<b></p><p>Levenshtein distance: <b>{result["death_date"]["levenshtein_distance"]}<b></p>{table_header}'
    #     output += f'<tr><td>{result["death_date"]["local"]}</td><td>{result["death_date"]["external"]}</td><td>{result["death_date"]["levenshtein_distance"]}</td></tr>'
    #     output += '</table>'

    # if 'death_place' in result:
    #     output += f'<h2>Place of death:</h2><p>Score: <b>{result["death_place"]["absolute_score"]} of {DEATH_PLACE_MAX_SCORE_CONTRIBUTION}<b></p><p>Mean Levenshtein ratio: <b>{result["death_place"]["mean_levenshtein_ratio_normalized"]} ({result["death_place"]["mean_levenshtein_ratio_original"]})<b></p>{table_header}'
    #     for value_pair in result['death_place']['matched_pairs']:
    #         output += f'<tr><td>{value_pair["local"]}</td><td>{value_pair["external"]}</td><td>{value_pair["levenshtein_ratio_normalized"]} ({value_pair["levenshtein_ratio_original"]})</td></tr>'
    #     output += '</table>'

    return output


#>1941-01-01