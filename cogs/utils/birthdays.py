"""Birthdays: a user's, from About Me, and a character's, stored with its persona.

Stored as `{"month": int, "day": int}`, plus `"year"` for a character only. A user gives no
year: a birthday is something a character can mention, and a year would turn it into an
age claim the bot has no business holding. A character's year is fiction and gives it an
age to turn.

Pure date logic, no I/O. `birthday_context_lines` is what reaches a prompt, and only for
the day before, the day of and the day after -- each judged on the clock of whoever has
the birthday.
"""
import datetime
from typing import Any, Dict, Optional

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")

_MONTH_LOOKUP = {name.lower(): i for i, name in enumerate(MONTH_NAMES, start=1)}
_MONTH_LOOKUP.update({name[:3].lower(): i for i, name in enumerate(MONTH_NAMES, start=1)})
_MONTH_LOOKUP["sept"] = 9

#: The longest each month runs in any year. February allows the 29th: a leap-day
#: birthday is real, and is observed on the 28th in other years.
_MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def parse_birthday(day: str, month: str, year: str = "", *,
                   allow_year: bool) -> Optional[Dict[str, int]]:
    """A stored birthday from what was typed, or None when every box was left blank.

    Day and month are separate boxes, so 03/04 cannot be read the wrong way round. A month
    is a number or a name. Raises ValueError with wording fit to show the user.
    """
    day, month, year = (day or "").strip(), (month or "").strip(), (year or "").strip()
    if not day and not month and not year:
        return None
    if not day or not month:
        raise ValueError("A birthday needs both a day and a month. Leave every box blank to clear it.")

    if month.isdigit():
        month_num = int(month)
    else:
        month_num = _MONTH_LOOKUP.get(month.lower().rstrip("."), 0)
    if not 1 <= month_num <= 12:
        raise ValueError(f"'{month}' is not a month. Use 1-12 or a name, such as 3 or March.")

    if not day.isdigit() or not 1 <= int(day) <= _MONTH_DAYS[month_num - 1]:
        raise ValueError(f"{MONTH_NAMES[month_num - 1]} has no day '{day}'.")
    birthday = {"month": month_num, "day": int(day)}

    if year:
        if not allow_year:
            raise ValueError("A year is not asked for here.")
        if not year.isdigit() or not 1 <= int(year) <= 9999:
            raise ValueError(f"'{year}' is not a year.")
        try:
            datetime.date(int(year), month_num, int(day))
        except ValueError:
            raise ValueError(f"{int(year)} was not a leap year, so it had no 29 February.")
        birthday["year"] = int(year)
    return birthday


def valid_birthday(value: Any) -> Optional[Dict[str, int]]:
    """`value` if it is a well-formed stored birthday, else None. Stored data is not trusted."""
    if not isinstance(value, dict):
        return None
    month, day, year = value.get("month"), value.get("day"), value.get("year")
    if not (isinstance(month, int) and 1 <= month <= 12
            and isinstance(day, int) and 1 <= day <= _MONTH_DAYS[month - 1]):
        return None
    if year is not None and not (isinstance(year, int) and 1 <= year <= 9999):
        return None
    return value


def format_birthday(value: Any) -> Optional[str]:
    """'14 March', or '14 March 1990'; None for anything that is not a birthday."""
    birthday = valid_birthday(value)
    if not birthday:
        return None
    text = f"{birthday['day']} {MONTH_NAMES[birthday['month'] - 1]}"
    return f"{text} {birthday['year']}" if birthday.get("year") else text


def _occurrence(birthday: Dict[str, int], year: int) -> datetime.date:
    """The date the birthday falls on in `year`. A leap-day birthday falls on the 28th."""
    month, day = birthday["month"], birthday["day"]
    if month == 2 and day == 29:
        try:
            return datetime.date(year, 2, 29)
        except ValueError:
            return datetime.date(year, 2, 28)
    return datetime.date(year, month, day)


def birthday_offset(value: Any, today: datetime.date, reach: int = 1) -> Optional[int]:
    """-1 if the birthday was yesterday, 0 today, 1 tomorrow; None for any other day.

    `reach` widens the window: a birthday within a day on its owner's clock is within two
    on any other, so `reach=2` rules a date out before that clock is looked up.

    Checked against the occurrences either side of the new year, so a 31 December birthday
    is still yesterday on 1 January.
    """
    birthday = valid_birthday(value)
    if not birthday:
        return None
    for year in (today.year - 1, today.year, today.year + 1):
        offset = (_occurrence(birthday, year) - today).days
        if -reach <= offset <= reach:
            return offset
    return None


def describe_birthday(value: Any, today: datetime.date, name: Optional[str] = None) -> Optional[str]:
    """One sentence for a prompt, or None when the birthday is not within a day of today.

    `name` None means the character's own birthday, said in the second person, with the
    age it turns when a year is known.
    """
    birthday = valid_birthday(value)
    offset = birthday_offset(birthday, today)
    if birthday is None or offset is None:
        return None
    when = {-1: "Yesterday was", 0: "Today is", 1: "Tomorrow is"}[offset]

    if name is not None:
        return f"{when} the birthday of {name}."

    sentence = f"{when} your birthday."
    if birthday.get("year"):
        occurred = today + datetime.timedelta(days=offset)
        age = occurred.year - birthday["year"]
        if age > 0:
            verb = {-1: "You turned", 0: "You turn", 1: "You turn"}[offset]
            sentence += f" {verb} {age}{' tomorrow' if offset == 1 else ''}."
    return sentence
