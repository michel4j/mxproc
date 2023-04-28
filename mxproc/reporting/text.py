import subprocess

from prettytable import PrettyTable


def plot(data, plot_type='linespoints', style='full-height'):
    raw_data = [data['x']] + data['y1']
    y2_start = len(raw_data)
    raw_data.extend(data.get('y2', []))
    data_array = list(zip(*raw_data))
    labels = data_array[0]

    height = 43 if style == 'full-height' else 24
    if data.get('x-scale') == 'inv-square':
        x_spec = '($1**-2)'
        extras = [
            'set xtics nomirror',
            'set link x via 1/x**2 inverse 1/x**0.5',
            'set x2tics'
        ]
    else:
        x_spec = '1'
        extras = []

    plots = [
        "'-' using {}:{} title '{}' with {} axes x1y1 ".format(x_spec, i + 2, labels[i], plot_type)
        for i in range(y2_start)
    ]
    plots += [
        "'-' using {}:{} title '{}' with {} axes x1y2 ".format(x_spec, i + 2, labels[i], plot_type)
        for i in range(y2_start, len(raw_data))
    ]
    plot_data = '\n'.join([
        '\t '.join(['{}'.format(val) for val in row]) for row in data_array[1:]
    ])
    commands = "\n".join(extras + [
        "set term dumb 110 {}".format(height), "plot " + ',\n\t'.join(plots),  plot_data,  "exit"
    ])

    process = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, errors = process.communicate(commands.encode('utf8'))
    return output  # .decode('utf-8')


def text_report(report):
    output = []
    for i, section in enumerate(report):
        if i != 0:
            output.append('\n\n{}\n\n'.format('-' * 79))
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
                        table.field_names = ['{}'.format(j) for j in range(len(content['data'][0]))]
                        for j, row in enumerate(content['data']):
                            table.add_row(row)
                            table.align['{}'.format(j)] = 'l' if j == 0 else 'r'

                    output.append(table.get_string())
                elif content.get('kind') in ['lineplot', 'scatterplot']:
                    plot_type = {'lineplot': 'linespoints', 'scatterplot': 'points'}[content['kind']]
                    plot_text = plot(content['data'], plot_type=plot_type,
                                     style=content.get('style', 'full-height'))
                    output.append(plot_text.decode('utf8'))
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
        return '\n{}\n{}'.format(text.title(), underline * len(text))
    else:
        return '\n{} {}'.format('#' * level, text)
