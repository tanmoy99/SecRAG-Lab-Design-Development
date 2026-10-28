import datetime as dt

import pytest

from secrag.eval.matching import contains_answer, extract_dates, extract_money, normalise


@pytest.mark.parametrize("text", ["It costs AUD 450.", "A$450 per year", "Fee: $450.00", "450 dollars", "aud 450"])
def test_money_forms_match(text):
    assert contains_answer(text, "money", "AUD 450")


@pytest.mark.parametrize("text", ["It costs $4,500.", "$45", "AUD 450,000", "$450.50", "450 people"])
def test_money_does_not_match_other_values(text):
    assert not contains_answer(text, "money", "AUD 450")


def test_money_with_thousands_separator():
    assert extract_money("Fee AUD 1,285 or $1285.00") == {1285}
    assert contains_answer("costs $1285", "money", "AUD 1,285")


@pytest.mark.parametrize("text", ["Monday 15 March 2027", "15/03/2027", "March 15, 2027", "2027-03-15",
                                  "15th Mar 2027"])
def test_date_forms_match(text):
    assert contains_answer(text, "date", "2027-03-15")


def test_numeric_dates_are_day_first():
    assert extract_dates("03/04/2027") == {dt.date(2027, 4, 3)}
    assert not contains_answer("16 March 2027", "date", "2027-03-15")


@pytest.mark.parametrize("text", ["Call (03) 5550 1234", "03 5550 1234", "+61 3 5550 1234", "03-5550-1234"])
def test_phone_forms_match(text):
    assert contains_answer(text, "phone", "(03) 5550 1234")


def test_phone_different_area_code_does_not_match():
    assert not contains_answer("(02) 5550 1234", "phone", "(03) 5550 1234")


def test_email_is_case_insensitive():
    assert contains_answer("Write to Privacy@Northbridge.example.", "email", "privacy@northbridge.example")
    assert not contains_answer("privacy@northbridge.example.com", "email", "privacy@northbridge.example")


@pytest.mark.parametrize("text", ["within 21 days", "twenty-one days", "a 21-day window", "21 calendar days"])
def test_number_with_unit_matches(text):
    assert contains_answer(text, "number", "21", "days", ("21 days", "twenty-one days"))


@pytest.mark.parametrize("text", ["21 business days", "21 weeks", "121 days", "the 21st"])
def test_number_needs_its_unit(text):
    assert not contains_answer(text, "number", "21", "days", ("21 days", "twenty-one days"))


def test_bare_number_is_not_an_answer():
    assert not contains_answer("there are 8 of them", "number", "8", "weeks", ("8 weeks", "eight weeks"))


def test_percent_forms():
    for text in ["8%", "8 per cent", "eight per cent", "8 percent"]:
        assert contains_answer(text, "number", "8", "percent", ("8%", "8 per cent"))


def test_text_alias_whole_word_only():
    assert contains_answer("Our Privacy Officer is Mei Tran.", "text", "Mei Tran")
    assert not contains_answer("Mei Tranter", "text", "Mei Tran")
    assert contains_answer("opens at 8.30am", "text", "8:30 am", aliases=("8:30 am", "8.30am"))


def test_normalise_unifies_quotes_dashes_and_space():
    assert normalise("O’Connor –  x y") == "o'connor - x y"
