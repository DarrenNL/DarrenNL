import datetime
import time
from xml.dom import minidom
import pandas as pd

def daily_readme(date):
    """
    Returns the length of time since inception date
    e.g. 'XX' days
    """
    return '{}'.format((datetime.datetime.today() - date).days)

def formatter(query_type, difference, funct_return=False, whitespace=0):
    """
    Prints a formatted time differential
    Returns formatted result if whitespace is specified, otherwise returns raw result
    """
    print('{:<23}'.format('   ' + query_type + ':'), sep='', end='')
    print('{:>12}'.format('%.4f' % difference + ' s ')) if difference > 1 else print('{:>12}'.format('%.4f' % (difference * 1000) + ' ms'))
    if whitespace:
        return f"{'{:,}'.format(funct_return): <{whitespace}}"
    return funct_return

def format_plural(unit):
    """
    Returns a properly formatted number
    e.g.
    'day' + format_plural(diff.days) == 5
    >>> '5 days'
    'day' + format_plural(diff.days) == 1
    >>> '1 day'
    """
    return 's' if unit != 1 else ''

def perf_counter(funct, *args):
    """
    Calculates the time it takes for a function to run
    Returns the function result and the time differential
    """
    start = time.perf_counter()
    funct_return = funct(*args)
    return funct_return, time.perf_counter() - start

def _get_element_by_id(node, uid):
    """Find element by id (minidom getElementById fails without DTD)."""
    if node.nodeType == node.ELEMENT_NODE and node.getAttribute('id') == uid:
        return node
    for child in node.childNodes:
        found = _get_element_by_id(child, uid)
        if found:
            return found
    return None


def svg_overwrite(filename, metrics):
    """
    Parse SVG files and update elements by ID.
    metrics: dict with keys cagr, days, total, alpha, beta, sharpe, drawdown, sortino, var95, cvar95, winrate
    """
    svg = minidom.parse(filename)
    ids = ['cagr', 'days', 'total', 'alpha', 'beta', 'sharpe', 'drawdown', 'sortino', 'var95', 'cvar95', 'winrate']
    for uid in ids:
        if uid in metrics:
            el = _get_element_by_id(svg.documentElement, uid)
            if el and el.firstChild:
                el.firstChild.data = str(metrics[uid])
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(svg.toxml('utf-8').decode('utf-8'))

if __name__ == '__main__':
    print('Calculation times:')
    age_data, age_time = perf_counter(daily_readme, datetime.datetime(2022, 7, 1))
    formatter('age calculation', age_time)
    data = pd.read_csv('performance.csv', header=None, index_col=0)[1].to_dict()
    metrics = {
        'cagr': "{}%".format(data.get('cagr', round(((data['total_performance']/100+1)**(365/int(age_data))-1)*100, 2))),
        'days': str(age_data),
        'total': "{}%".format(data['total_performance']),
        'alpha': str(data.get('alpha', '')),
        'beta': str(data.get('beta', '')),
        'sharpe': str(data.get('sharpe', '')),
        'drawdown': "{}%".format(data.get('roll_1y', '')) if data.get('roll_1y') not in (None, '') else "N/A",
        'sortino': str(data.get('sortino', '')),
        'cvar95': "{}%".format(data.get('cvar_95', '')),
        'winrate': "{}%".format(data.get('win_rate', '')),
    }
    svg_overwrite('dark_ver.svg', metrics)
    svg_overwrite('light_ver.svg', metrics)
