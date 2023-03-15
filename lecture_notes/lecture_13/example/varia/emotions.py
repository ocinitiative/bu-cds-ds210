"""Package with functions for operating on emotions in strings.

Not very useful and doesn't handle that many real-world cases.
"""

# note that __EMOTIONS__ and _set_emotions do not make it to the library
__EMOTIONS__ = ['happy','sad','angry','surprised']

def remove_emotions(description):
    """Remove prefixes corresponding to emotions.

    Args:
        descripion (str): The string to be processed

    Returns:
        str: The input string without prefixes corresponding to emotions if there are any.
    """

    global __EMOTIONS__
    for prefix in __EMOTIONS__:
        if description.startswith(prefix + ' '):
            # recurse in case there is another emotion
            return remove_emotions(description[len(prefix)+1:])
    return description

def _set_emotion(description, emotion, capitalize):
    tmp = emotion + ' ' + remove_emotions(description)
    return tmp.capitalize() if capitalize else tmp

def make_angry(description, capitalize = False):
    """Make the described object **angry.**

    This is an example of a Google Docstring formatting.

    Args:
        description (str): The string to be processed
        capitalize (bool): Whether the first character in the output should 
            be capitalized (default is False)

    Returns:
        str: The input string without initial emotions and with the prefix `angry` instead.
    """
    return _set_emotion(description, 'angry', capitalize)

def make_surprised(description, capitalize = False):
    """Make the described object **surprised.**

    This is an example of a numpy doc formatting.

    Parameters
    ----------
    description: str
        The string to be processed
    capitalize: bool, optional
        Whether the first character in the output should be capitalized (default is False)

    Returns
    -------
    str
        The input string without initial emotions and with the prefix `surprised` instead.
    """
    return _set_emotion(description, 'surprised', capitalize)

def make_happy(description, capitalize = False):
    """Make the described object **happy.**

    This is an example of a Google Docstring formatting.

    Args:
        description (str): The string to be processed
        capitalize (bool): Whether the first character in the output should 
            be capitalized (default is False)

    Returns:
        str: The input string without initial emotions and with the prefix `happy` instead.
    """
    return _set_emotion(description, 'happy', capitalize)

def make_sad(description, capitalize = False):
    """Make the described object **sad.**

    This is an example of a numpy doc formatting.

    Parameters
    ----------
    description: str
        The string to be processed
    capitalize: bool, optional
        Whether the first character in the output should be capitalized (default is False)

    Returns
    -------
    str
        The input string without initial emotions and with the prefix `sad` instead.
    """
    return _set_emotion(description, 'sad', capitalize)
