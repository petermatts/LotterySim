def timesSinceLastDraw(data: dict) -> dict:
    """
    Returns a mapped dictionary

    Maps frequency list into int of times since last draw

    Input is a dict of str -> list of lists
        string is signifyer for lotto ball type

    TODO be clearer        
    """

    result = {}
    keys = data.keys()

    for k in keys:
        result_d = {}
        x = 1
        for i in data[k]:
            tmp = i
            tmp.reverse()
            result_d[str(x)] = tmp.index(1)
            x += 1
        result[k] = result_d

    return result

        
