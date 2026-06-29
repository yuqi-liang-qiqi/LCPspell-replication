"""
@Author  : Yuqi Liang 梁彧祺
@File    : method_labels.py
@Time    : 30/01/2026 14:00
@Desc    :
Map internal distance-method keys (as returned by simulations) to short display
strings for tables and optional helper plots. This is not Matplotlib-specific;
pipeline figure scripts may use shorter axis labels than this helper.
"""

from __future__ import annotations


def method_display_name(method_raw: str) -> str:
    """Map internal method keys to publication display names."""
    if method_raw == 'HAM':
        return 'HAM'
    if method_raw in {'LCP', 'RLCP', 'LCPmst', 'RLCPmst', 'OM'}:
        return method_raw

    if method_raw.startswith('LCPspell_expcost_'):
        p = method_raw.replace('LCPspell_expcost_', '')
        return f'LCPspell({p})'
    if method_raw.startswith('RLCPspell_expcost_'):
        p = method_raw.replace('RLCPspell_expcost_', '')
        return f'RLCPspell({p})'
    if method_raw.startswith('OMspell_expcost_'):
        p = method_raw.replace('OMspell_expcost_', '')
        return f'OMspell({p})'
    if method_raw.startswith('OMspellRS_expcost_'):
        p = method_raw.replace('OMspellRS_expcost_', '')
        return f'OMspellRS({p})'

    return method_raw
