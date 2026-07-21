import itertools
import math
import os

import plotille
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


class AutoTickFormatter:
    ticks: np.ndarray
    limits: tuple[float, float]

    def __init__(self, minimum, maximum, num_ticks=8, default=True):
        self.min = minimum
        self.max = maximum
        self.digits = 2
        self.num_ticks = num_ticks
        self.default = default
        self.update_limits()

    def update_limits(self):
        """Generates clean, human-readable axis ticks."""
        if self.min == self.max:
            self.min = self.min - 1
            self.max = self.max + 1

        # Find the raw step size
        range_val = self.max - self.min
        raw_step = range_val / (self.num_ticks - 1)

        # Calculate a "nice" step size (1, 2, or 5 * 10^x)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        normalized = raw_step / magnitude

        if normalized < 1.5:
            step = 1 * magnitude
        elif normalized < 3:
            step = 2 * magnitude
        elif normalized < 7:
            step = 5 * magnitude
        else:
            step = 10 * magnitude

        # Snap the min and max to the new step
        nice_min = math.floor(self.min / step) * step
        nice_max = math.ceil(self.max / step) * step

        # Generate the ticks, rounding to prevent floating-point drift
        ticks = []
        current = nice_min
        decimals = max(0, -int(math.floor(math.log10(step))))

        while current <= nice_max + (step * 0.1):
            ticks.append(round(current, decimals))
            current += step

        self.ticks = np.array(ticks)
        self.digits = decimals
        self.limits = float(nice_min), float(nice_max)
        return self.limits

    def get_limits(self):
        return self.limits

    def get_value(self, min_, max_):
        if self.default:
            return round(min_, self.digits)
        elif np.abs(self.ticks - min_).min() < (max_ - min_)/2:
            return round(min_, self.digits)
        else:
            return np.nan

    def __call__(self, min_, max_):
        value = self.get_value(min_, max_)
        if np.isnan(value) or np.isinf(value):
            return ''
        return f'{value:g}'


class InvSqrFormatter(AutoTickFormatter):

    def get_value(self, min_, max_):
        if self.default:
            return round(min_, self.digits)
        elif np.abs(self.ticks - min_).min() < (max_ - min_)/2:
            return round(min_, self.digits)
        else:
            return np.nan


def float_or_nan(val):
    """
    Convert to float or return NaN.
    :param val:
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def plot(data, plot_type='lineplot', style='full-height'):
    """
    Generates a text-based plot using plotille.
    """

    os.environ['COLORTERM'] = 'truecolor'

    cmap = plt.get_cmap('Set1')

    x_label = data['x'][0]
    # convert to float, but handle potential non-numeric values gracefully
    x_values = np.array(list(map(float_or_nan, data['x'][1:])))
    x_tick_fmt = AutoTickFormatter(np.nanmin(x_values), np.nanmax(x_values))

    if data.get('x-scale') == 'inv-square':
        # plotille doesn't support non-linear scales, so we transform the data.
        # The axis will show the transformed values.
        with np.errstate(divide='ignore'):  # ignore divide by zero
            x_values = 1 / (x_values ** 2)

        x_tick_fmt = InvSqrFormatter(np.nanmin(x_values), np.nanmax(x_values))
    series = [
        [s for s in data.get(ax_name)]
        for ax_name in ['y1', 'y2']
        if ax_name in data
    ]

    output = ""
    color_index = 0
    for group in series:

        all_y_values = np.array(list(itertools.chain.from_iterable([line[1:] for line in group]))).astype(float)
        y_min = np.nanmin(all_y_values)
        y_max = np.nanmax(all_y_values)
        y_tick_fmt = AutoTickFormatter(y_min, y_max, default=False)

        fig = plotille.Figure()
        fig.y_ticks_fkt = y_tick_fmt
        fig.x_ticks_fkt = x_tick_fmt
        fig.color_mode = 'rgb'
        fig.origin = False
        fig.width = 80
        fig.height = 24 if style == 'full-height' else 16
        fig.x_label = x_label
        fig.set_x_limits(*x_tick_fmt.get_limits())
        fig.set_y_limits(*y_tick_fmt.get_limits())

        for line in group:
            rgba_color = cmap(color_index)
            hex_color = to_hex(rgba_color).strip('#')   # remove leading hash
            color_index += 1

            y_label = line[0]
            y_values = np.array(list(map(float_or_nan, line[1:])))

            if plot_type == 'lineplot':
                fig.plot(x_values, y_values, label=y_label, lc=hex_color)
            else:
                fig.scatter(x_values, y_values, label=y_label, lc=hex_color)

        output += fig.show(legend=True)
    os.environ.pop("COLORTERM", None)
    return output


def text_report(report):
    output = []
    for i, section in enumerate(report):
        if i != 0:
            output.append(f'\n\n{"-" * 79}\n\n')
        output.append(heading(section['title'], 1))
        if 'description' in section:
            output.append(section['description'])
        if 'content' in section:
            for content in section['content']:
                if 'title' in content:
                    output.append(heading(content['title'], 2))
                output.append(content.get('description', ''))
                if content.get('kind') == 'table':
                    table = PrettyTable()
                    if content.get('header') == 'row':
                        table.field_names = content['data'][0]
                        for row in content['data'][1:]:
                            table.add_row(row)
                        table.align = 'r'
                    else:
                        table.header = False
                        table.field_names = [f'{j}' for j in range(len(content['data'][0]))]
                        for j, row in enumerate(content['data']):
                            table.add_row(row)
                            table.align[f'{j}'] = 'l' if j == 0 else 'r'

                    output.append(table.get_string())
                elif content.get('kind') in ['lineplot', 'scatterplot']:
                    try:
                        plot_text = plot(
                            content['data'],
                            plot_type=content['kind'],
                            style=content.get('style', 'full-height')
                        )
                    except ValueError:
                        plot_text = "\n\n! Error Generating Text Plot !\n\n"
                    output.append(plot_text)
                if 'notes' in content:
                    output.append(heading('NOTES', 4))
                    output.append(content['notes'] + '\n')
        if 'notes' in section:
            output.append(heading('NOTES', 3))
            output.append(section['notes'] + '\n')

    return '\n'.join(output)


def heading(text, level):
    if level in [1, 2]:
        underline = {1: '=', 2: '-'}[level]
        return f'\n{text.title()}\n{underline * len(text)}'
    else:
        return f'\n{"#" * level} {text}'
