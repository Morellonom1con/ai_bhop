"""
string_manager.py

Functions which deal with strings
"""

from difflib import ndiff

__all__ = []


def split_string_around_stars(string):
    """It splits the provided string around every star in it
    and returns the result in the form of a list

    Input:
    ------
    string : string

    Output:
    -------
    splitted_string : list
    """

    splitted_string = []
    word = ""

    for char in string:
        if char == '*':
            if word:
                splitted_string.append(word)
                word = ""
            splitted_string.append('*')
        else:
            word += char

    if word:
        splitted_string.append(word)

    return splitted_string


def string_minus_string(string1, string2):
    """It returns the strings we have to add to "string1" to get it directly
    equal to "string2" in cases in which "string2" contains "string1" by parts
    plus some differences

    Input:
    ------
    string1 : string
    string2 : string

    Output:
    -------
    differences : list of strings
    """

    differences = []
    chars_differences = list(ndiff(string1, string2))
    chars_differences = list(enumerate([[position, char_diff[-1]]
                                        for position, char_diff in
                                        enumerate(chars_differences)
                                        if char_diff[0] == "+"]))

    if chars_differences:
        differences.append(chars_differences[0][1][1])
        for ttuple in chars_differences[1:]:
            index = ttuple[0]
            position = ttuple[1][0]
            character = ttuple[1][1]
            last_position = chars_differences[index-1][1][0]
            if position == last_position+1:
                differences[-1] += character
            else:
                differences.append(character)

    return differences


def replace_stars_in_string(string, str_list):
    """It replaces every star in "string" by its associated string
    in "str_list" and returns the new string created

    Input:
    ------
    string : string
    str_list : list of strings

    Output:
    -------
    new_string : string
    """

    splitted_string = split_string_around_stars(string)
    index = 0

    for position, sstring in enumerate(splitted_string):
        if sstring == '*':
            splitted_string[position] = str_list[index]
            index += 1

    new_string = "".join(splitted_string)

    return new_string
