from typing import Dict
from types import MappingProxyType


symbols = MappingProxyType(
    {
        "set_empty": "∅",
        "set_union": "∪",
        "set_intersection": "∩",
        "set_is_element_of": "∈",
        "set_is_not_element_of": "∉",
        "set_is_subset_of": "⊆",
        "set_is_proper_subset_of": "⊂",
        "set_is_not_subset_of": "⊄",
        "set_is_superset_of": "⊇",
        "set_is_proper_superset_of": "⊃",
        "set_is_not_superset_of": "⊅",
        "set_universal": "ξ",
        "set_minus": "∖",
        "set_cartesian_product": "×",
        "set_difference": "\\",
        "set_symmetric_difference": "Δ",
        "set_cardinality": "|",
        "set_complement": "'",
        "set_power_set": "ℙ",
        "set_contains_as_member": "∋",
        "set_does_not_contain_as_member": "∌",
        "complex_numbers": "ℂ",
        "real_numbers": "ℝ",
        "rational_numbers": "ℚ",
        "integers": "ℤ",
        "natural_numbers": "ℕ",
        "superscript_i": "ⁱ",
        "superscript_n": "ⁿ",
        "superscript_plus_sign": "⁺",
        "superscript_minus": "⁻",
        "superscript_zero_0": "⁰",
        "superscript_one_1": "¹",
        "superscript_two_2": "²",
        "superscript_three_3": "³",
        "superscript_four_4": "⁴",
        "superscript_five_5": "⁵",
        "superscript_six_6": "⁶",
        "superscript_seven_7": "⁷",
        "superscript_eight_8": "⁸",
        "superscript_nine_9": "⁹",
        "subscript_zero_0": "₀",
        "subscript_one_1": "₁",
        "subscript_two_2": "₂",
        "subscript_three_3": "₃",
        "subscript_four_4": "₄",
        "subscript_five_5": "₅",
        "subscript_six_6": "₆",
        "subscript_seven_7": "₇",
        "subscript_eight_8": "₈",
        "subscript_nine_9": "₉",
        "fraction_one_half_1_2": "½",
        "fraction_zero_thirds_0_3": "↉",
        "fraction_one_third_1_3": "⅓",
        "fraction_two_thirds_2_3": "⅔",
        "fraction_one_quarter_1_4": "¼",
        "fraction_three_quarters_3_4": "¾",
        "fraction_one_fifth_1_5": "⅕",
        "fraction_two_fifths_2_5": "⅖",
        "fraction_three_fifths_3_5": "⅗",
        "fraction_four_fifths_4_5": "⅘",
        "fraction_one_sixth_1_6": "⅙",
        "fraction_five_sixths_5_6": "⅚",
        "fraction_one_seventh_1_7": "⅐",
        "fraction_one_eighth_1_8": "⅛",
        "fraction_three_eighths_3_8": "⅜",
        "fraction_five_eighths_5_8": "⅝",
        "fraction_seven_eighths_7_8": "⅞",
        "fraction_one_ninth_1_9": "⅑",
        "fraction_one_tenth_1_10": "⅒",
        "degree_angle": "°",
        "degree_celsius": "℃",
        "degree_fahrenheit": "℉",
        "diameter": "⌀",
        "multiplication": "✕",
        "division": "÷",
        "left_floor": "⌊",
        "right_floor": "⌋",
        "left_ceiling": "⌈",
        "right_ceiling": "⌉",
        "product": "∏",
        "coproduct": "∐",
        "summation": "∑",
        "summation_top": "⎲",
        "summation_bottom": "⎳",
        "partial_differential": "∂",
        "integral": "∫",
        "top_half_integral": "⌠",
        "bottom_half_integral": "⌡",
        "not_sign": "¬",
        "plus_minus_sign": "±",
        "square_root": "√",
        "cube_root": "∛",
        "fourth_root": "∜",
        "proportional_to": "∝",
        "infinity": "∞",
        "right_angle": "∟",
        "angle": "∠",
        "therefore": "∴",
        "approximately_equal_to": "≅",
        "not_equal_to": "≠",
        "less_than_or_equal_to": "≤",
        "greater_than_or_equal_to": "≥",
        "right_angle_with_arc": "⊾",
        "right_triangle": "⊿",
        "dot_operator": "⋅",
    }
)


def search(namelike: str) -> Dict:

    """
    Return a dictionary of symbols with names
    containing namelike argument
    """    

    filtered = {k: v for k, v in symbols.items() if namelike in k}

    return filtered


def print_symbols(symbols_to_print:Dict = symbols) -> None:

    """
    Prints symbols in the symbols argument,
    or all if this is omitted
    """

    for symbol in symbols_to_print.keys():

        print(f"{symbol.ljust(32, ' ')}", end="")

        print(f" {symbols_to_print[symbol]}", end="")

        print(f"   decimal: {str(ord(symbols_to_print[symbol])).ljust(5, ' ')}", end="")

        print(f"   hexadecimal: \\u{hex(ord(symbols_to_print[symbol]))[2:]}")