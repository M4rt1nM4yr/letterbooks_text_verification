def count_characters(s):
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    sorted_char_count = dict(sorted(char_count.items(), key=lambda item: item[1], reverse=True))
    return sorted_char_count
