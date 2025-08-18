from functools import partial
from typing import Iterable, Dict, Any


class TableFormatter:

    def __init__(
            self,
            column_width: int = 8,
            label_width: int = 12,
            float_precision: int = 4,
            alignment: str = '>',
    ):
        self.column_width = column_width
        self.label_width = label_width
        self.float_precision = float_precision
        self.alignment = alignment

        self.label_formatter = format_one_label
        self.value_formatter = partial(
            format_one_value,
            width=column_width,
            float_precision=float_precision,
            alignment=alignment,
        )

        self.output = ""
        self.column_names = []

    def result(self):
        """Get formated table"""
        return self.output

    def header(self, header_label: str, column_names: Iterable[str]):
        """Create header"""
        self.column_names = column_names
        header_str = f'\t {header_label:{self.label_width}s} \t'
        for name in column_names:
            header_str += f'{name:{self.alignment}{self.column_width}s} \t'
        self._amend(header_str)

    def row_ordered(self, label: str, values: Iterable):
        """Make one new row, with ordered values"""
        row_str = f'\t {label:{self.label_width}s} \t'
        for value in values:
            row_str += self.value_formatter(value)
        self._amend(row_str)

    def row_unordered(self, label: str, values: Dict[str, Any]):
        """Make one new row, with unordered values"""
        # reorder
        ordered_values = []
        for column in self.column_names:
            ordered_values.append(values[column])
        # delegate
        self.row_ordered(label, ordered_values)

    def content(self, values: Dict[Any, Dict[str, Any]]):
        """Make all rows"""
        for label, line in values.items():
            self.row_unordered(self.label_formatter(label), line)

    def new_line(self):
        self.output += '\n'

    def _amend(self, line):
        self.output += (line + '\n')


def format_one_value(value, width: int = 8, float_precision: int = 4, alignment: str = '>') -> str:
    if isinstance(value, float):
        return f'{value:{alignment}{width}.{float_precision}f} \t'
    elif isinstance(value, int):
        return f'{value:{alignment}{width}d} \t'
    else:
        return f'{str(value):{alignment}{width}s} \t'


def format_one_label(label, label_format: str = "{:s}") -> str:
    return label_format.format(label)
